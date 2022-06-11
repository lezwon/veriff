import abc
import asyncio
import os
import time
from abc import abstractmethod
from typing import List, Optional, cast, Dict, Tuple

import cv2
import httpx
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.ops.gen_nn_ops import TopKV2

from app.constants import model_url, labels_url
from app.logger import Logger

logger = Logger.getLogger(__name__, True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable Tensorflow logging


class Classifier(abc.ABC):
    def __init__(
        self,
        model_url: str,
        labels_url: str,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        concurrency: int = 5,
    ) -> None:
        self.model = self.load_model(model_url)
        self.labels = self.load_labels(labels_url)
        self.input_shape = input_shape
        self.client = httpx.AsyncClient(timeout=10.0)
        self.concurrency = concurrency
        self.semaphore = None

        self.warmup()  # Warm up the model

    def load_model(self, model_url: str) -> hub.KerasLayer:
        logger.info("Loading model")
        return hub.KerasLayer(model_url)

    @abstractmethod
    def load_labels(self, labels_url: str) -> List[str]:
        raise NotImplementedError

    def warmup(self):
        """
        Warm Up the model with some sample tensors.
        :return: None
        """
        logger.info("Warming up model")
        # Create a random float tensor of size (4, 224, 244)
        image_tensor = tf.random.uniform((4,) + self.input_shape)
        # Call the model on the tensor in a loop 10 times
        for _ in range(10):
            self.model.call(image_tensor)

    def download(self, url: str) -> bytes:
        """
        Download a file from a URL.
        :param url: URL to download from
        :return: Bytes of the file
        """
        logger.info(f"Downloading {url}")
        response = httpx.get(labels_url)
        return response.read()

    async def download_async(self, url: str) -> bytes:
        """
        Download a file from a URL.
        :param url: URL to download from
        :return: Bytes of the file
        """
        async with self.semaphore:
            logger.info(f"Downloading {url}")
            response = await self.client.get(url)

        return await response.aread()

    def predict(self, image_tensor: tf.Tensor) -> tf.Tensor:
        """
        Infers the given tensor.
        :param image_urls: List of image urls
        :return: List of predictions
        """
        logger.info("Inferring Images")
        return self.model.call(image_tensor)

    def initialize_semaphore(self):
        return asyncio.Semaphore(self.concurrency)


class BirdClassifier(Classifier):
    """
    Classifier class for Birds
    """

    def __init__(
        self,
        model_url: str,
        labels_url: str,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        concurrency: int = 5,
    ):
        """
        Initialize the classifier
        """
        super().__init__(model_url, labels_url, input_shape, concurrency)

    def load_labels(self, labels_url: str) -> List[str]:
        """
        Read and parse labels into a list
        :return:
        """
        logger.info("Loading labels")
        bird_labels_lines = self.download(labels_url).decode().strip().split("\n")
        bird_labels_lines.pop(0)  # remove header (id, name)
        mapped_list = map(lambda bird_line: bird_line.split(","), bird_labels_lines)
        labels = sorted(mapped_list, key=lambda bird_line: bird_line[0])
        return list(map(lambda bird_line: bird_line[1], labels))

    async def preprocess(self, image_url: str) -> Optional[np.ndarray]:
        """
        Download and preprocess images
        :param image_url:
        :return:
        """

        try:
            # Loading images
            image_array = np.asarray(
                bytearray(await self.download_async(image_url)), dtype=np.uint8
            )
            # Changing images
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Generate tensor
            return tf.convert_to_tensor(image, dtype=tf.float32)
        except httpx.HTTPError as e:
            logger.error(f"Failed to download image {image_url}")
            logger.error(type(e))
            return None
        except Exception as e:
            logger.error(f"Failed to process image {image_url}")
            logger.error(type(e))
            return None

    def _get_top_n_scores(self, model_raw_output: tf.Tensor, k: int = 3) -> TopKV2:
        """
        Get the scores and indices of the top k scores
        :param model_raw_output: tf.Tensor of shape (b, num_classes)
        :param k: Number of top scores to return
        :return: TopKV2
        """
        return tf.math.top_k(model_raw_output, k=k)

    def post_process(
        self, model_raw_output: tf.Tensor, k: int
    ) -> List[List[Dict[str, str]]]:
        """
        Postprocess the model output.
        :param model_raw_output: tf.Tensor of shape (b, num_classes)
        :return: List of predictions
        """
        scores, indices = self._get_top_n_scores(model_raw_output, k)
        results = []

        for image_index, (im_scores, im_indices) in enumerate(
            zip(scores.numpy(), indices.numpy()), start=1
        ):
            image_results = []
            for label_index, score in zip(im_indices, im_scores):
                image_results.append(
                    {"bird_name": self.labels[label_index], "score": float(score)}
                )

            results.append(image_results)

        return results

    def filter_urls(
        self, image_list: List[Optional[tf.Tensor]], image_urls: List[str]
    ) -> Tuple[List, List]:
        """
        Filter out the failed and successful urls
        :param image_list: List of images
        :param image_urls: List of image urls
        :return: Tuple of failed and successful urls
        """
        successful_urls, failed_urls = [], []
        # Pick image_urls based on image_list
        for i, image in enumerate(image_list):
            if image is not None:
                successful_urls.append(image_urls[i])
            else:
                failed_urls.append(image_urls[i])
        return failed_urls, successful_urls

    async def run(
        self, image_urls: List[str], k=3
    ) -> Tuple[Dict[str, List], List[str]]:
        """
        Runs the classifier
        :param image_urls: List of image urls
        :return: List of predictions
        """
        self.semaphore = self.initialize_semaphore()

        image_list = await asyncio.gather(
            *[self.preprocess(url) for i, url in enumerate(image_urls)]
        )
        # Remove empty values
        filtered_list = [image for image in image_list if image is not None]
        # Convert to tensor
        image_tensor = tf.stack(filtered_list) / 255
        # Run inference
        model_raw_output = self.predict(image_tensor)
        failed_urls, successful_urls = self.filter_urls(
            cast(List[Optional[tf.Tensor]], image_list), image_urls
        )

        # Postprocess
        k_top_results = self.post_process(model_raw_output, k)
        # Return results
        successful_inference = dict(zip(successful_urls, k_top_results))
        return successful_inference, failed_urls


if __name__ == "__main__":
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg",
        "https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg",
        "https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg",
        "https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg",
    ]

    start_time = time.time()
    classifier = BirdClassifier(model_url, labels_url)
    asyncio.run(classifier.run(image_urls))
    print("Time spent: %s" % (time.time() - start_time))
