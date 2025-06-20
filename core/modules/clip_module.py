import torch
import numpy as np
from typing import List
from PIL import Image
from imgdata.imgdata.structure import ImageObject
from base import EnricherModule
from model_config import ModelLoader

class ClipModule(EnricherModule):
    def __init__(self, model=None, processor=None, label_texts=None, device="cuda"):
        model_bundle = ModelLoader().get_model("clip")
        self.model = model or model_bundle["model"]
        self.processor = processor or model_bundle["processor"]
        self.label_texts = label_texts or model_bundle["label_texts"]
        self.device = device

        with torch.no_grad():
            self.label_features = self._encode_texts(self.label_texts)

    def parse(self, objects: List[ImageObject], **kwargs) -> List[ImageObject]:
        for obj in objects:
            image = obj.mask_image if obj.mask_image is not None else obj.image
            label, score = self._classify(image)
            obj.label = label
            obj.score = score
            obj.source_module = 'clip'
        return objects

    def _classify(self, image: np.ndarray):
        image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = torch.matmul(image_features, self.label_features.T)
            best_idx = similarity.argmax().item()
            best_score = similarity[0, best_idx].item()
            return self.label_texts[best_idx], round(float(best_score), 4)

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

if __name__ == "__main__":
    clipModule = ClipModule()
    from PIL import Image
    image = np.array(Image.open('/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image1.png'))
    result = clipModule.parse([ImageObject(image)])
    print(result)
