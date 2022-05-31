from unittest import mock
from unittest.mock import MagicMock

import constants
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


def invalid_content():
    return "Content not Found"


def invalid_image():
    image = tf.random.uniform(shape=(244, 311, 3), minval=0, maxval=255, dtype=tf.int32)
    image = tf.bitcast(tf.cast(image, tf.int8), tf.uint8)
    # convert image to bytes
    return image.numpy().tobytes()


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
        (
            [
                "https://example.com/8c3d7060.jpg",
            ],
            0,
        ),
        (
            [
                "https://example.com/8c3d7060.jpg",
            ],
            10,
        ),
    ],
)
@mock.patch("api.classifier.client.get", new_callable=AsyncMock)
def test_successful_inference(mock_async_client, image_bytes, image_urls, k):
    """
    Test that the model output is similar to the expected output.
    """
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


@pytest.mark.parametrize(
    ["image_urls", "k", "expected_error"],
    [
        ([], 1, "No images provided"),
        (
            ["https://example.com/8c3d7060.jpg"] * 11,
            3,
            f"Too many images provided. Max {constants.max_images} images allowed",
        ),
        (
            ["https://example.com/8c3d7060.jpg"],
            -1,
            f"Min allowed value of K is {constants.min_k}",
        ),
        (
            ["https://example.com/8c3d7060.jpg"] * 10,
            111,
            f"Max allowed value of K is {constants.max_k}",
        ),
    ],
)
def test_input_validations(image_urls, k, expected_error):
    """
    Test that the input image urls and k are validated.
    """
    data = {"images": image_urls, "k": k}

    response = client.post("/", json=data)
    assert response.status_code == 200, "Invalid Response Received"
    data = response.json()
    assert len(data["successful_results"]) == 0, "Invalid number of successful results"
    assert len(data["failed_results"]) == 0, "Invalid number of failed results"
    assert len(data["errors"]) == 1, "Invalid number of errors"
    assert data["errors"][0] == expected_error, "Invalid error message"


@pytest.mark.parametrize(
    ["image_url", "image"],
    [
        ("https://example.com/invalid_image.jpg", invalid_image),
        ("https://example.com/invalid_content.jpg", invalid_content),
    ],
)
@mock.patch("api.classifier.client.get", new_callable=AsyncMock)
def test_invalid_images(mock_async_client, image_bytes, image_url, image):
    """
    Test images which are not valid.
    """

    def side_effect_func(url):
        response = AsyncMock(name="Response")
        response.aread.return_value = image() if url == image_url else image_bytes
        return response

    mock_async_client.side_effect = side_effect_func

    image_urls = ["https://example.com/8c3d7060.jpg", image_url]
    data = {"images": image_urls, "k": 1}

    response = client.post("/", json=data)

    assert response.status_code == 200, "Invalid Response Received"
    data = response.json()
    assert (
        len(data["successful_results"]) == len(image_urls) - 1
    ), "Unexpected number of successful results"
    assert len(data["failed_results"]) == 1, "Unexpected number of failed results"
    assert len(data["errors"]) == 0, "Unexpected number of errors"
    assert data["failed_results"][0] == image_url, "Invalid image url in failed result"


@mock.patch("api.classifier.client.get", new_callable=AsyncMock)
def test_grayscale_images(mock_async_client, grayscale_image):
    """
    Test grayscale images.
    """
    response = AsyncMock(name="Response")
    response.aread.return_value = grayscale_image

    mock_async_client.return_value = response

    image_urls = ["https://example.com/8c3d7060.jpg"]
    data = {"images": image_urls, "k": 1}

    response = client.post("/", json=data)

    assert response.status_code == 200, "Invalid Response Received"
    data = response.json()
    assert len(data["successful_results"]) == len(
        image_urls
    ), "Unexpected number of successful results"
    assert len(data["failed_results"]) == 0, "Unexpected number of failed results"
    assert len(data["errors"]) == 0, "Unexpected number of errors"
