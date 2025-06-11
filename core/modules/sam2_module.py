import numpy as np
from typing import List
from imgdata.structure import ImageObject
from base import BaseModule

class SamModule(BaseModule):
    def __init__(self, sam_model):
        self.model = sam_model

    def parse(self, image: np.ndarray, **kwargs) -> List[ImageObject]:
        masks = self.model.generate(image)
        return [ImageObject.from_mask(mask=m) for m in masks]


# import torch
# import numpy as np
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# import os

# class SAM2Segmenter:
#     def __init__(self,
#                  model_cfg = os.environ.get('sam2_config_file'),
#                  checkpoint= os.environ.get('sam2_checkpoint_file')):
#         """
#         初始化 SAM2Segmenter 类。

#         Args:
#             model_cfg (str): 模型配置文件路径。
#             checkpoint (str): 模型检查点文件路径。
#         """
#         self.model = build_sam2(str(model_cfg), str(checkpoint))
#         self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
#         self.predictor = SAM2ImagePredictor(self.model)

#     def predict_one_instance(self, image, prompts=None):
#         """
#         使用 SAM2 模型进行预测，一次预测一类物体。

#         Returns:
#             mask
#         """
#         self.predictor.set_image(image)
#         with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#             if prompts is not None:
#                 masks, scores, _ = self.predictor.predict(**prompts)
#             else:
#                 masks, scores, _ = self.predictor.predict()

#         max_index = np.argmax(scores)
#         self.mask = masks[max_index]
#         return self.mask

#     def predict_whole_image(self, image):
#         """
#         Generates masks for the given image.

#         Arguments:
#           image (np.ndarray): The image to generate masks for, in HWC uint8 format.

#         Returns:
#            list(dict(str, any)): A list over records for masks. Each record is
#              a dict containing the following keys:
#                segmentation (dict(str, any) or np.ndarray): The mask. If
#                  output_mode='binary_mask', is an array of shape HW. Otherwise,
#                  is a dictionary containing the RLE.
#                bbox (list(float)): The box around the mask, in XYWH format.
#                area (int): The area in pixels of the mask.
#                predicted_iou (float): The model's own prediction of the mask's
#                  quality. This is filtered by the pred_iou_thresh parameter.
#                point_coords (list(list(float))): The point coordinates input
#                  to the model to generate this mask.
#                stability_score (float): A measure of the mask's quality. This
#                  is filtered on using the stability_score_thresh parameter.
#                crop_box (list(float)): The crop of the image used to generate
#                  the mask, given in XYWH format.
#         """
#         return self.mask_generator.generate(image)
