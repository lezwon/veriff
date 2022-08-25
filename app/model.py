import torch


class YoloModel:
    def __init__(self):
        # Model
        self.model = torch.hub.load(
            "ultralytics/yolov5", "yolov5s"
        )  # or yolov5n - yolov5x6, custom

    def infer(self, img):
        # Inference
        results = self.model(img)
        return results
        # results.save()
