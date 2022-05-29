import asyncio
import os
import time

import cv2
import httpx
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable Tensorflow logging

model_url = "https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
labels_url = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv"

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg",
    "https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg",
    "https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg",
    "https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg",
]


class BirdClassifier:
    """
    Classifier class for Birds
    """

    @staticmethod
    def load_model():
        """
        Loads the model using Keras
        :return:
        """
        # TODO: Put in GPU
        return hub.KerasLayer(model_url)

    def load_and_cleanup_labels(self):
        """
        Read and parse labels into a list
        :return:
        """
        bird_labels_raw = httpx.get(labels_url)
        bird_labels_lines = bird_labels_raw.read().decode().strip().split("\n")
        bird_labels_lines.pop(0)  # remove header (id, name)
        birds = [None] * len(bird_labels_lines)

        for bird_line in bird_labels_lines:
            bird_id, bird_name = bird_line.split(",")
            birds[int(bird_id)] = bird_name

        return birds

    async def preprocess(self, image_url):
        """
        Download and preprocess images
        :param image_url:
        :return:
        """
        # TODO: Write tests
        # TODO: handle images that dont exist

        # Loading images
        image_get_response = await self.client.get(image_url)
        print(f"Download {image_url}")
        # image_get_response = urllib.request.urlopen(image_url)
        image_array = np.asarray(
            bytearray(await image_get_response.aread()), dtype=np.uint8
        )
        # Changing images
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Generate tensor
        return tf.convert_to_tensor(image, dtype=tf.float32)

    async def main(self):
        """
        Main function to run the code
        :return:
        """
        self.bird_model = self.load_model()
        self.bird_labels = self.load_and_cleanup_labels()
        self.client = httpx.AsyncClient()

        image_list = await asyncio.gather(
            *[self.preprocess(url) for i, url in enumerate(image_urls)]
        )
        image_tensor = tf.stack(image_list) / 255
        # TODO: Put image in GPU

        model_raw_output = self.bird_model.call(image_tensor)

        # TODO: Print results to kubernetes log
        scores, indices = self.get_top_n_scores(model_raw_output)

        for image_index, (im_scores, im_indices) in enumerate(
            zip(scores.numpy(), indices.numpy()), start=1
        ):
            print(f"Run: {image_index}")

            print(
                f'Top match: "{self.bird_labels[im_indices[0]]}" with score: {im_scores[0]}'
            )
            print(
                f'Second match: "{self.bird_labels[im_indices[1]]}" with score: {im_scores[1]}'
            )
            print(
                f'Third match: "{self.bird_labels[im_indices[2]]}" with score: {im_scores[2]}'
            )
            print("\n")

    def get_top_n_scores(self, model_raw_output, k=3):
        """
        Get the scores and indices of the top k scores
        :param model_raw_output:
        :param k:
        :return:
        """
        return tf.math.top_k(model_raw_output, k=k)


if __name__ == "__main__":
    start_time = time.time()
    classifier = BirdClassifier()
    asyncio.run(classifier.main())
    print("Time spent: %s" % (time.time() - start_time))
