from core.imgdata.image_data import BBox, ImageParseItem, ImageParseResult
from core.modules.module_factory import ModuleFactory
from core.pipeline.custom_omni_parser import CustomOmniParser
from core.pipeline.semantic_parser import SemanticParser
import numpy as np
from PIL import Image

if __name__ == "__main__":
    # clip
    clipModule = ModuleFactory.get_module("clip")
    from PIL import Image

    image = np.array(Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image1.png"))
    result = clipModule.parse([ImageParseItem(image, "", 0, None)], filter="image")
    print(result)

    # # florence2
    # florence2Module = ModuleFactory.get_module("florence2_icon")
    # image = np.array(Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image1.png"))
    # result = florence2Module.parse([ImageParseItem(image, "", 0, None)], filter="image")
    # print(result)

    # # paddleocr
    # paddleocr_module = ModuleFactory.get_module("paddleocr")

    # image = np.array(Image.open(r"/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg"))
    # result: ImageParseResult = paddleocr_module.parse(image)
    # print(result)

    # # sam2
    # sam_module = ModuleFactory.get_module("sam2")

    # image = np.array(Image.open(r"/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg"))
    # result: ImageParseResult = sam_module.parse(image)
    # print(result)

    # # yolo
    # yoloModule = ModuleFactory.get_module("yolo")

    # image = Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB")
    # image_np = np.array(image)

    # result = yoloModule.parse(image_np)
    # print(result)

    # # CustomOmniParser
    # omni_parser = CustomOmniParser()

    # image = Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB")
    # image_np = np.array(image)

    # result = omni_parser.parse(image_np)
    # print(result)

    # # SemanticParser
    # semantic_parser = SemanticParser()

    # image = Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB")
    # image_np = np.array(image)

    # result = semantic_parser.parse(image_np)
    # print(result)
