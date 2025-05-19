import torch
from transformers import (
    AutoProcessor,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification
)
import os

class MaskCategoryPredictor:
    """
    封装了基于 BLIP-2 的区域描述和基于 CLIP 的零样本分类。

    参数:
        blip_model_name (str): BLIP-2 模型名称，例如 "Salesforce/blip2-opt-2.7b"。
        clip_model_name (str): CLIP 模型名称，例如 "openai/clip-vit-base-patch32"。
        device (str): 设备，"cuda" 或 "cpu"。
    """
    def __init__(
        self,
        blip_model_name: str = "Salesforce/blip2-opt-2.7b",
        clip_processor_path: str = os.environ.get('clip-vit-base-patch32-processor'),
        clip_model_path: str = os.environ.get('clip-vit-base-patch32-model'),
        device: str = "cuda"
    ):
        # 加载 BLIP-2
        self.device = device
        self.blip_processor = AutoProcessor.from_pretrained(blip_model_name)
        self.blip_model = (
            AutoModelForVisualQuestionAnswering.from_pretrained(blip_model_name)
                .to(device)
        )
        # 加载 CLIP
        self.clip_processor = AutoProcessor.from_pretrained(clip_processor_path)
        self.clip_model = (
            AutoModelForZeroShotImageClassification.from_pretrained(clip_model_path)
                .to(device)
        )

    def crop_regions(self, image: torch.Tensor, masks: list) -> list:
        """
        根据 masks 将原图裁剪成多个区域。

        参数:
            image (Tensor HxWx3): 原始图像（BGR 或 RGB）。
            masks (list of dict): 每个 dict 含 key "segmentation" 对应 Bool 数组。

        返回:
            List of PIL.Image or Tensor 区域列表。
        """
        regions = []
        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            ys, xs = mask.nonzero()
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()
            region = image[ymin:ymax+1, xmin:xmax+1, :]
            regions.append(region)
        return regions

    def describe_regions(self, regions: list, max_length: int = 40) -> list:
        """
        使用 BLIP-2 为每个区域生成自由描述。

        参数:
            regions (list): Image or numpy array 列表。
            max_length (int): 文本最大长度。

        返回:
            List of str, 每个区域的描述。
        """
        descriptions = []
        for region in regions:
            inputs = self.blip_processor(images=region, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_length=max_length)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            descriptions.append(caption)
        return descriptions

    def classify_regions(self, regions: list, categories: list) -> list:
        """
        基于 CLIP 的零样本图像分类。

        参数:
            regions (list): Image or numpy array 列表。
            categories (list of str): 候选类别标签。

        返回:
            List of str, 每个区域最可能的类别。
        """
        texts = [f"a photo of {c}" for c in categories]
        # 文本编码
        text_inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_feats = self.clip_model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)

        predictions = []
        for region in regions:
            image_inputs = self.clip_processor(images=region, return_tensors="pt").to(self.device)
            with torch.no_grad():
                img_feats = self.clip_model.get_image_features(**image_inputs)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            # 相似度计算
            logits = (img_feats @ text_feats.T).squeeze(0)
            idx = logits.argmax().item()
            predictions.append(categories[idx])
        return predictions

    def predict(
        self,
        image: torch.Tensor,
        masks: list,
        categories: list = None,
        max_length: int = 40
    ) -> dict:
        """
        对 masks 中的每个区域进行描述和/或分类。

        参数:
            image (Tensor HxWx3): 原始图像。
            masks (list): SAM2AutomaticMaskGenerator 输出。
            categories (list of str, optional): 如果提供，则使用 classify_regions；否则仅返回描述。
            max_length (int): 描述文本最大长度。

        返回:
            dict:
              "regions": list of 裁剪区域,
              "descriptions": list of 描述文本,
              "categories": list of 预测类别 或 None
        """
        regions = self.crop_regions(image, masks)
        descriptions = self.describe_regions(regions, max_length)
        result = {"regions": regions, "descriptions": descriptions}
        if categories:
            categories_pred = self.classify_regions(regions, categories)
            result["categories"] = categories_pred
        else:
            result["categories"] = None
        return result

if __name__ == "__main__":
    # Usage 示例:
    predictor = MaskCategoryPredictor()
    from zeroshot_segm import SAM2Segmenter

    segm = SAM2Segmenter()
    import numpy as np
    from PIL import Image

    image = np.array(Image.open(r'E:\xingchen\memory_data\images\2025-05-06_21-20-00.856.png').convert('RGB'))

    masks = segm.predict_whole_image(image)
    result = predictor.predict(image, masks)
    print(result["descriptions"])
