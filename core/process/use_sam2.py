import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import xc_config

class SAM2Segmenter:
    def __init__(self,
                 model_cfg = xc_config._xc_configs_dir / "sam2.1/sam2.1_hiera_1.yaml",
                 checkpoint= xc_config._xc_checkpoints_dir / "sam2.1/sam2.1_hiera_large.pt"):
        """
        初始化 SAM2Segmenter 类。

        Args:
            model_cfg (str): 模型配置文件路径。
            checkpoint (str): 模型检查点文件路径。
        """
        self.model = build_sam2(str(model_cfg), str(checkpoint))
        self.predictor = SAM2ImagePredictor(self.model)
        self.prompts = None
        self.mask = None

    def set_image(self, image: np.array):
        self.predictor.set_image(image)

    def set_prompts(self, prompts):
        self.prompts = prompts

    def predict(self):
        """
        使用 SAM2 模型进行预测，一次预测一类物体。

        Returns:
            mask
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.prompts:
                masks, scores, _ = self.predictor.predict(**self.prompts)
            else:
                masks, scores, _ = self.predictor.predict()

        max_index = np.argmax(scores)
        self.mask = masks[max_index]
        return self.mask
