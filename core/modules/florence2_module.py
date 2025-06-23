# core/modules/florence2_module.py
import torch
import numpy as np
import cv2
from torchvision.transforms import ToPILImage
from typing import List, Optional
from PIL import Image  # 缺少这一行
from imgdata.imgdata.structure import ImageObject
from base import EnricherModule
from model_config import ModelLoader


class Florence2Module(EnricherModule):
    def __init__(self, model=None, processor=None, device="cuda"):
        model_bundle = ModelLoader().get_model("florence2")
        self.model = model or model_bundle["model"]
        self.processor = processor or model_bundle["processor"]
        self.device = device

    def parse(self, objects: List[ImageObject], **kwargs) -> List[ImageObject]:
        to_pil = ToPILImage()
        prompt = "<CAPTION>"
        for obj in objects:
            # 使用 mask_image 优先，否则用原图
            image = obj.mask_image if obj.mask_image is not None else obj.image
            image = to_pil(image).convert("RGB")

            # processor 返回 tokenized 图像
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
                self.device, dtype=torch.float16
            )

            # 生成式图像理解
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    temperature=0.7,
                    min_length=16,
                    max_length=128,
                )

            # 解码为文本
            text = self.processor.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            obj.text = text

        return objects

    # @torch.inference_mode()
    # def parse(self, objects: List[ImageObject], starting_idx: int = 0) -> List[ImageObject]:
    #     to_pil = ToPILImage()
    #     prompt = "<CAPTION>"

    #     for obj in objects[starting_idx:]:
    #         image = obj.mask_image if obj.mask_image is not None else obj.image
    #         image = cv2.resize(image, (64, 64))

    #         pil_image = to_pil(image).convert("RGB")

    #         inputs = self.processor(
    #             images=pil_image,
    #             text=prompt,
    #             return_tensors="pt",
    #             do_resize=False # You're already resizing with cv2.resize
    #         ).to(device=self.device, dtype=torch.float16)

    #         generated_ids = self.model.generate(
    #             input_ids=inputs["input_ids"],
    #             pixel_values=inputs["pixel_values"],
    #             max_new_tokens=20,
    #             num_beams=1,
    #             do_sample=False
    #         )

    #         # Decode to text
    #         text = self.processor.tokenizer.decode(
    #             generated_ids[0], skip_special_tokens=True
    #         )
    #         obj.text = text

    #     return objects


if __name__ == "__main__":
    florence2Module = Florence2Module()
    from PIL import Image

    image = np.array(Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image1.png"))
    result = florence2Module.parse([ImageObject(image)])
    print(result)
