import torch
import numpy as np
from typing import List
from imgdata.imgdata.structure import ImageObject, ImageParseResult
from base import BaseModule
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SamModule(BaseModule):
    def __init__(self, sam_model):
        self.model = sam_model
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
        self.predictor = SAM2ImagePredictor(self.model)

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        res = self.mask_generator.generate(image)
        return ImageParseResult.from_sam2(image, res)

    def parse_with_prompts(self, image: np.ndarray, prompts=None, **kwargs) -> ImageObject:
        self.predictor.set_image(image)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if prompts is not None:
                masks, scores, _ = self.predictor.predict(**prompts)
            else:
                masks, scores, _ = self.predictor.predict()

        max_index = np.argmax(scores)
        mask = masks[max_index]
        return ImageObject.from_mask(image, mask)
