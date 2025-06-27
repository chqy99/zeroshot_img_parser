from core.imgdata.image_data import BBox, ImageParseItem, ImageParseResult
from core.modules.module_factory import ModuleFactory
from core.imgtools import visualizer, html_visualizer
import numpy as np
from PIL import Image

if __name__ == "__main__":
    # # clip
    # import core.modules.clip_module
    # clipModule = ModuleFactory.get_module("clip")
    # from PIL import Image

    # image = np.array(Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image1.png"))
    # result = clipModule.parse([ImageParseItem(image, "", 0, None)], filter="image")
    # print(result)

    # # florence2
    # import core.modules.florence2_module
    # florence2Module = ModuleFactory.get_module("florence2_icon")
    # image = np.array(Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image1.png"))
    # result = florence2Module.parse([ImageParseItem(image, "", 0, None)], filter="image")
    # print(result)

    # paddleocr
    # import core.modules.paddleocr_module
    # paddleocr_module = ModuleFactory.get_module("paddleocr")

    # image = np.array(Image.open(r"/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB"))
    # result: ImageParseResult = paddleocr_module.parse(image)
    # # print(result)
    # # vis_img = visualizer.visualize_parse_result(result, show_mask=False, show_bbox=True)
    # # vis_img.save("./paddleocr_bbox_overlay.png")
    # # print("保存完成: paddleocr_bbox_overlay.png")
    # html_content = html_visualizer.generate_html_for_result(result)
    # with open("parse_result_visualization.html", "w", encoding="utf-8") as f:
    #     f.write(html_content)
    # print("HTML 文件已保存，打开查看")

    # sam2
    import core.modules.sam2_module
    sam_module = ModuleFactory.get_module("sam2")

    image = np.array(Image.open(r"/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg"))
    result: ImageParseResult = sam_module.parse(image)
    # # print(result)
    # vis_img = visualizer.visualize_parse_result(result, show_mask=True, show_bbox=False)
    # vis_img.save("image_sam2_vis.png")
    html_content = html_visualizer.generate_html_for_result(result)
    with open("parse_result_visualization.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("HTML 文件已保存，打开查看")

    # yolo
    # import core.modules.yolo_module
    # yoloModule = ModuleFactory.get_module("yolo")

    # image = Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB")
    # image_np = np.array(image)

    # result = yoloModule.parse(image_np)
    # # print(result)
    # vis_img = visualizer.visualize_parse_result(result, show_mask=False, show_bbox=True)
    # vis_img.save("image_yolo_vis.png")

    # # CustomOmniParser
    # from core.pipeline.custom_omni_parser import CustomOmniParser
    # omni_parser = CustomOmniParser()

    # image = Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB")
    # image_np = np.array(image)

    # result = omni_parser.parse(image_np)
    # print(result)

    # # SemanticParser
    # from core.pipeline.semantic_parser import SemanticParser
    # semantic_parser = SemanticParser()

    # image = Image.open("/MLU_OPS/DEV_SOFT_TRAIN/chenqiyang/image.jpg").convert("RGB")
    # image_np = np.array(image)

    # result = semantic_parser.parse(image_np)
    # print(result)
