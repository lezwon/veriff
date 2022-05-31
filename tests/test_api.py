from unittest import mock
from unittest.mock import MagicMock

import pytest
import tensorflow as tf
from api import app
from fastapi.testclient import TestClient

client = TestClient(app)


class AsyncMock(MagicMock):  # Not needed if using Python 3.8
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


# Fixture to generate image bytes
@pytest.fixture
def image_bytes():
    image = tf.random.uniform(
        shape=(1024, 1024, 3), minval=0, maxval=255, dtype=tf.int32
    )
    image = tf.bitcast(tf.cast(image, tf.int8), tf.uint8)
    # convert image to jpg
    image_data = tf.io.encode_jpeg(image)
    # convert image to bytes
    return image_data.numpy()


@pytest.fixture
def invalid_content():
    return "Content not Found"


@pytest.fixture
def invalid_image():
    image = tf.random.uniform(shape=(244, 311, 3), minval=0, maxval=255, dtype=tf.int32)
    image = tf.bitcast(tf.cast(image, tf.int8), tf.uint8)
    # convert image to bytes
    return image.numpy().to_bytes()


@pytest.fixture
def grayscale_image():
    image = tf.random.uniform(
        shape=(1000, 1000, 1), minval=0, maxval=255, dtype=tf.int32
    )
    image = tf.bitcast(tf.cast(image, tf.int8), tf.uint8)
    # convert image to jpg
    image_data = tf.io.encode_jpeg(image)
    # convert image to bytes
    return image_data.numpy()


@pytest.mark.parametrize(
    ["image_urls", "k"],
    [
        (
            [
                "https://example.com/8c3d7060.jpg",
                "https://example.com/8c3d7061.jpg",
                "https://example.com/8c3d7062.jpg",
            ],
            5,
        ),
        (
            [
                "https://example.com/8c3d7060.jpg",
                "https://example.com/8c3d7061.jpg",
                "https://example.com/8c3d7062.jpg",
                "https://example.com/8c3d7063.jpg",
                "https://example.com/8c3d7064.jpg",
            ],
            3,
        ),
    ],
)
@mock.patch("api.classifier.client.get", new_callable=AsyncMock)
def test_inference(mock_async_client, image_bytes, image_urls, k):
    response = AsyncMock(name="Response")
    response.aread.return_value = image_bytes
    mock_async_client.return_value = response

    data = {"images": image_urls, "k": k}

    response = client.post("/", json=data)

    assert response.status_code == 200, "Invalid Response Received"
    data = response.json()
    assert len(data["successful_results"]) == len(
        image_urls
    ), "Invalid number of successful results"
    assert len(data["failed_results"]) == 0, "Invalid number of failed results"
    assert len(data["errors"]) == 0, "Invalid number of errors"
    assert (
        len(next(iter(data["successful_results"].values()))) == k
    ), "Invalid number of k predictions"


# TODO: test invalid images
# TODO: test empty array
# TODO: test invalid k
# TODO: Test more than 10 images
# TODO: Invalid URL
