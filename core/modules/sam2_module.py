import torch
import numpy as np
from typing import List
from imgdata.imgdata.image_parse import BBox, ImageParseItem, ImageParseResult
from base import BaseModule
from model_config import ModelLoader
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SamModule(BaseModule):
    def __init__(self, sam_model=None):
        self.model = sam_model or ModelLoader().get_model("sam2")
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
        self.predictor = SAM2ImagePredictor(self.model)

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        res = self.mask_generator.generate(image)
        parse_res = ImageParseResult(image=image)
        for item in res:
            mask = item["segmentation"]
            bbox_input = item["bbox"]
            bbox = BBox(
                bbox_input[0],
                bbox_input[1],
                bbox_input[0] + bbox_input[2],
                bbox_input[1] + bbox_input[3],
            )
            score = item["stability_score"]
            parse_res.items.append(
                ImageParseItem(
                    image=image, source_module="sam2", score=score, bbox=bbox, mask=mask, type='instance'
                )
            )
        return parse_res

    def parse_with_prompts(
        self, image: np.ndarray, prompts=None, **kwargs
    ) -> ImageParseItem:
        self.predictor.set_image(image)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if prompts is not None:
                masks, scores, _ = self.predictor.predict(**prompts)
            else:
                masks, scores, _ = self.predictor.predict()

        max_index = np.argmax(scores)
        bbox = BBox.mask_to_bbox(masks[max_index])
        return ImageParseItem(
            image, "sam2", scores[max_index], bbox, masks[max_index], type='instance'
        )


if __name__ == "__main__":
    sam_module = SamModule()

    from PIL import Image

    image = np.array(Image.open(r"/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg"))
    result: ImageParseResult = sam_module.parse(image)
    print(result)
