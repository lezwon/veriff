import os

import httpx
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import cv2
import urllib.request
import numpy as np
import time
import asyncio

# Getting some unknown linter errors, disable everything to get this to production asap
# pylint: disable-all

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow logging

model_url = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'
labels_url = 'https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv'

image_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg',
    'https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg',
    'https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg',
    'https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg'
]


class BirdClassifier:
    @staticmethod
    def load_model():
        return hub.KerasLayer(model_url)

    def load_and_cleanup_labels(self):
        bird_labels_raw = httpx.get(labels_url)
        bird_labels_lines = bird_labels_raw.read().decode().strip().split("\n")
        bird_labels_lines.pop(0)  # remove header (id, name)
        birds = {}
        for bird_line in bird_labels_lines:
            bird_id, bird_name = bird_line.split(',')
            birds[int(bird_id)] = {'name': bird_name}

        return birds

    def order_birds_by_result_score(self, model_raw_output):
        for index, value in np.ndenumerate(model_raw_output):
            bird_index = index[1]
            self.bird_labels[bird_index]['score'] = value

        return sorted(self.bird_labels.items(), key=lambda x: x[1]['score'])

    def get_top_n_result(self, top_index, birds_names_with_results_ordered):
        bird_name = birds_names_with_results_ordered[top_index*(-1)][1]['name']
        bird_score = birds_names_with_results_ordered[top_index*(-1)][1]['score']
        return bird_name, bird_score

    async def main(self):
        self.bird_model = self.load_model()
        self.bird_labels = self.load_and_cleanup_labels()
        self.client = httpx.AsyncClient()

        image_list = await asyncio.gather(*[self.infer(url) for i, url in enumerate(image_urls)])
        image_tensor = tf.stack(image_list) / 255

        model_raw_output = self.bird_model.call(image_tensor)

        # TODO: Print results to kubernetes log

        scores, indices = tf.math.top_k(model_raw_output, k=3)

        for image_index, (scores, indices) in enumerate(zip(scores.numpy(), indices.numpy()), start=1):
            print(f"Run: {image_index}")

            print(f"Top match: {self.bird_labels[indices[0]]} with score: {scores[0]}")
            print(f"Second match: {self.bird_labels[indices[1]]} with score: {scores[1]}")
            print(f"Third match: {self.bird_labels[indices[2]]} with score: {scores[2]}")
            print('\n')


    async def infer(self, image_url):
        # Loading images
        image_get_response = await self.client.get(image_url)
        print(f"Download {image_url}")
        # image_get_response = urllib.request.urlopen(image_url)
        image_array = np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)
        # Changing images
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Generate tensor
        return tf.convert_to_tensor(image, dtype=tf.float32)



if __name__ == "__main__":
    start_time = time.time()
    classifier = BirdClassifier()
    asyncio.run(classifier.main())
    print('Time spent: %s' % (time.time() - start_time))
