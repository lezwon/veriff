import abc
import asyncio
import os
from abc import abstractmethod
from typing import List, Tuple

import httpx
import tensorflow as tf
import tensorflow_hub as hub

from app.constants import LABELS_URL
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
        connect_timeout: int = 10,
    ) -> None:
        """
        Initialize the classifier
        :param model_url: TFHub URL to the model
        :param labels_url: URL to the labels file
        :param input_shape: Shape of the input tensor
        :param concurrency: Number of concurrent downloads allowed. Defaults to 5
        :param connect_timeout: Timeout for the HTTP connection. Defaults to 10
        """

        self.model = self.load_model(model_url)
        self.labels = self.load_labels(labels_url)
        self.input_shape = input_shape
        self.client = httpx.AsyncClient(timeout=connect_timeout)
        self.concurrency = concurrency
        self.semaphore = None

        self.warmup()  # Warm up the model

    @staticmethod
    def load_model(model_url: str) -> hub.KerasLayer:
        """
        Load the model from a TFHub URL.
        :param model_url:
        :return:
        """
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

    @staticmethod
    def download(url: str) -> bytes:
        """
        Download a file from a URL
        :param url: URL to download from
        :return: Bytes of the file
        """
        logger.info(f"Downloading {url}")
        response = httpx.get(LABELS_URL)
        return response.read()

    async def download_async(self, url: str) -> bytes:
        """
        Download a file from a URL
        :param url: URL to download from
        :return: Bytes of the file
        """
        async with self.semaphore:
            logger.info(f"Downloading {url}")
            response = await self.client.get(url)

        return await response.aread()

    def predict(self, image_tensor: tf.Tensor) -> tf.Tensor:
        """
        Infers the given tensor
        :param image_tensor: Tensor to infer
        :return: List of predictions
        """
        logger.info("Inferring Images")
        return self.model.call(image_tensor)

    def initialize_semaphore(self):
        return asyncio.Semaphore(self.concurrency)
