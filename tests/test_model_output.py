import pytest
import tensorflow as tf
from app.bird_classifier import BirdClassifier
from app.constants import model_url, labels_url

# Set tf seed
tf.random.set_seed(42)


@pytest.mark.parametrize(
    ["images", "expected_scores", "expected_labels"],
    [
        (
            tf.random.normal((4, 224, 224, 3)),
            tf.constant([[0.9717838], [0.97310793], [0.9751507], [0.9728329]]),
            tf.constant([[964], [964], [964], [964]]),
        )
    ],
)
def test_model_output_is_similar(images, expected_scores, expected_labels):
    """
    Test that the model output is similar to the expected output.
    :param images: Tensor of float values
    :param expected_scores: 1D Tensor of expected float values
    :param expected_labels: 1D Tensor of expected int values
    :return:
    """
    classifier = BirdClassifier(model_url, labels_url)
    output = classifier.model.call(images)
    scores, labels = classifier._get_top_n_scores(output, k=1)
    assert tf.reduce_all(tf.math.equal(labels, expected_labels)), "Labels are not equal"

    # check scores with allclose
    assert tf.experimental.numpy.allclose(
        scores, expected_scores, rtol=1e-03, atol=5e-03, equal_nan=False
    ), "Scores do not match"
