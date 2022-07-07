import asyncio
import time
from typing import Tuple, List, Optional, Dict, cast

import cv2
import httpx
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import TopKV2

from app.classifier import Classifier, logger
from app.constants import MODEL_URL, LABELS_URL


class BirdClassifier(Classifier):
    """
    Classifier class for Birds
    """

    def __init__(
        self,
        model_url: str,
        labels_url: str,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        **kwargs,
    ):
        """
        Initialize the classifier
        """
        super().__init__(model_url, labels_url, input_shape, **kwargs)

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
        :param image_url: Image url
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

    @staticmethod
    def _get_top_n_scores(model_raw_output: tf.Tensor, k: int = 3) -> TopKV2:
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
        Postprocess the model output
        :param model_raw_output: tf.Tensor of shape (b, num_classes)
        :param k: Number of top scores to return
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

    @staticmethod
    def filter_urls(
        image_list: List[Optional[tf.Tensor]], image_urls: List[str]
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
        self, image_urls: List[str], k: int = 3
    ) -> Tuple[Dict[str, List], List[str]]:
        """
        Runs the classifier
        :param image_urls: List of image urls
        :param k: Number of top scores to return
        :return: List of predictions
        """
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
    IMAGE_URLS = [
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg",
        "https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg",
        "https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg",
        "https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg",
    ]
    start_time = time.time()
    classifier = BirdClassifier(MODEL_URL, LABELS_URL)
    successful, failed = classifier.run(IMAGE_URLS, k=3)
    print(successful)
    print(failed)
