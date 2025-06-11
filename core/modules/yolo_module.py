from ultralytics import YOLO
import numpy as np
from typing import List
from imgdata.structure import ImageObject
from base import BaseModule

class YoloModule(BaseModule):
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def parse(self, image: np.ndarray, **kwargs) -> List[ImageObject]:
        result = self.model.predict(image)[0]
        return [
            ImageObject.from_detection(bbox=box.xyxy[0], label=label, score=score)
            for box, label, score in zip(result.boxes, result.names, result.probs)
        ]