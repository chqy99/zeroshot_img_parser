# core/modules/florence2_module.py
import torch
import numpy as np
from typing import List
from PIL import Image  # 缺少这一行
from imgdata.imgdata.structure import ImageObject
from base import EnricherModule
from model_config import ModelLoader


class Blip2Module(EnricherModule):
    def __init__(self, model=None, processor=None, device="cuda"):
        model_bundle = ModelLoader().get_model("blip2")
        self.model = model or model_bundle["model"]
        self.processor = processor or model_bundle["processor"]
        self.device = device

    def ensure_rgb_pil(self, image):
        """确保输入为 RGB 格式的 PIL.Image"""
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("Unsupported image format")
        return image

    def parse(self, objects: List[ImageObject], **kwargs) -> List[ImageObject]:
        for obj in objects:
            # 使用 mask_image 优先，否则用原图
            image = obj.mask_image if obj.mask_image is not None else obj.image
            image = self.ensure_rgb_pil(image)

            # blip2 processor 返回 tokenized 图像
            inputs = self.processor(
                images=image, return_tensors="pt"
            ).to(self.device)

            # 如果模型是 float16，输入也转半精度 float16
            if getattr(self.model, "dtype", torch.float32) == torch.float16:
                inputs = {
                    k: v.half() if v.dtype == torch.float32 else v
                    for k, v in inputs.items()
                }

            # 生成式图像理解
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_length=128)

            # 解码为文本
            text = self.processor.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            obj.text = text

        return objects


if __name__ == "__main__":
    blip2Module = Blip2Module()
    from PIL import Image

    image = np.array(Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image1.png"))
    result = blip2Module.parse([ImageObject(image)])
    print(result)
