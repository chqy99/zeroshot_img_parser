from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
import numpy as np
from typing import List

from core.imgdata.image_data import BBox, ImageParseItem, ImageParseResult
from core.modules.base import BaseModule
from core.modules.model_config import ModelLoader
from core.modules.module_factory import ModuleFactory

@ModelLoader.register_loader("sam2")
def load_model_sam2(cfg, device):
    return build_sam2(cfg["model_cfg"], cfg["checkpoint"]).to(device)

class SamModule(BaseModule):
    def __init__(self, model):
        self.model = model
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
                    image=image,
                    source_module="sam2",
                    score=score,
                    bbox=bbox,
                    mask=mask,
                    type="instance",
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
            image, "sam2", scores[max_index], bbox, masks[max_index], type="instance"
        )

@ModuleFactory.register_module("sam2")
def build_module_sam2():
    model = ModelLoader().get_model("sam2")
    return SamModule(model)
