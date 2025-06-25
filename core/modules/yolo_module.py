from ultralytics import YOLO
import numpy as np
from typing import List
from imgdata.imgdata.image_parse import BBox, ImageParseItem, ImageParseResult
from base import BaseModule
from model_config import ModelLoader
from PIL import Image


class YoloModule(BaseModule):
    def __init__(self, model=None):
        self.model = model or ModelLoader().get_model("yolo")

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        # 确保输入是 RGB 3 通道
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]  # 去掉 alpha 通道
        elif image.ndim == 2:  # 灰度图 -> RGB
            image = np.stack([image] * 3, axis=-1)

        result = self.model.predict(image, verbose=False)[0]

        boxes = result.boxes
        names = result.names  # {0: 'icon'}

        parse_items: List[ImageParseItem] = []

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            label = names.get(cls_id, str(cls_id))  # 类别名字符串

            bbox = BBox(*xyxy)  # 假设 BBox 接受 4 个 float 值
            item = ImageParseItem(
                image, "yolo", score=conf, bbox=bbox, type="region", label=label
            )
            parse_items.append(item)

        return ImageParseResult(image=image, items=parse_items)


if __name__ == "__main__":
    yoloModule = YoloModule()

    # 加载图像并转换为 RGB 格式，再转 numpy
    image = Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB")
    image_np = np.array(image)

    result = yoloModule.parse(image_np)
    print(result)
