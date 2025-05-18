import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os

class SAM2Segmenter:
    def __init__(self,
                 model_cfg = os.environ.get('sam2_config_file'),
                 checkpoint= os.environ.get('sam2_checkpoint_file')):
        """
        初始化 SAM2Segmenter 类。

        Args:
            model_cfg (str): 模型配置文件路径。
            checkpoint (str): 模型检查点文件路径。
        """
        self.model = build_sam2(str(model_cfg), str(checkpoint))
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
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

    def predict_whole_image(self, image):
        masks = self.mask_generator.generate(image)
        return masks

import cv2

def visualize_masks_with_transparency(image, masks, alpha=1):
    # 创建结果图像（RGBA）
    vis_image = image.copy()
    overlay = np.zeros_like(vis_image, dtype=np.uint8)

    # 为每个 mask 分配一个颜色
    num_masks = len(masks)
    rng = np.random.default_rng(seed=42)
    colors = rng.integers(0, 255, size=(num_masks, 3))

    for i, mask_dict in enumerate(masks):
        mask = mask_dict["segmentation"]  # bool mask of shape (H, W)
        color = colors[i].tolist()  # [B, G, R]

        # 创建一个 mask 区域彩色图像
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask] = color

        # 累加到 overlay
        overlay = cv2.add(overlay, colored_mask)

    # 混合叠加（带透明度）
    cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)

    return vis_image

if __name__ == "__main__":
    segm = SAM2Segmenter()

    from PIL import Image

    image = np.array(Image.open(r'E:\xingchen\memory_data\images\2025-05-06_21-20-00.856.png').convert('RGB'))
    masks = segm.predict_whole_image(image)

    show_img = visualize_masks_with_transparency(image, masks)
    cv2.imwrite("segmentation_overlay.png", show_img)
