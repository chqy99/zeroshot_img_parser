# 组合 zeroshot_seg(sam2), ocr, zeroshot_classify, 多模态model 来预测 image_dataclass

from paddleocr import PaddleOCR
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, predict
from transformers import AutoProcessor, BlipForConditionalGeneration
import cv2
import torch
import numpy as np

# 加载模型
ocr = PaddleOCR(use_angle_cls=True, lang='ch')
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth").to("cuda")
predictor = SamPredictor(sam)

dino_model = load_model("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth")
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

# 加载图像
image = cv2.imread("your_image.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# OCR
ocr_results = ocr.ocr(image)
ocr_output = []
for line in ocr_results[0]:
    text, bbox = line[1][0], line[0]
    x0, y0 = int(bbox[0][0]), int(bbox[0][1])
    x1, y1 = int(bbox[2][0]), int(bbox[2][1])
    ocr_output.append({"text": text, "bbox": [x0, y0, x1, y1]})

# GroundingDINO
prompts = ["button", "icon", "person", "text", "logo"]
boxes, phrases = predict(
    model=dino_model,
    image=image_rgb,
    caption=", ".join(prompts),
    box_threshold=0.3,
    text_threshold=0.25
)

# SAM 分割 GroundingDINO 区域
predictor.set_image(image_rgb)
masks = []
for box in boxes:
    x0, y0, x1, y1 = map(int, box)
    mask, _, _ = predictor.predict(box=np.array([x0, y0, x1, y1]))
    masks.append(mask[0])

# BLIP 生成描述
def describe(region):
    inputs = blip_processor(images=region, return_tensors="pt").to("cuda")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# 构建输出结构
outputs = []

for i, (box, phrase, mask) in enumerate(zip(boxes, phrases, masks)):
    x0, y0, x1, y1 = map(int, box)
    region = image_rgb[y0:y1, x0:x1]
    description = describe(region)
    outputs.append({
        "source": "GroundingDINO",
        "class_name": phrase,
        "bbox": [x0, y0, x1, y1],
        "mask": mask.tolist(),
        "description": description
    })

for item in ocr_output:
    x0, y0, x1, y1 = item["bbox"]
    region = image_rgb[y0:y1, x0:x1]
    description = describe(region)
    outputs.append({
        "source": "OCR",
        "class_name": "text",
        "bbox": item["bbox"],
        "mask": None,
        "description": description + "（文字内容：" + item["text"] + "）"
    })

# 输出保存
import json
with open("parsed_result.json", "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)
