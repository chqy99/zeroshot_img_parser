from paddleocr import PaddleOCR
import numpy as np

def points_to_bbox(points):
    """
    将多个点转换为边界框

    参数:
        points: 一个包含多个点的列表，每个点是一个包含x和y坐标的列表或元组
                例如: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    返回:
        一个列表形式的边界框，格式为 [x_min, y_min, x_max, y_max]
    """
    # 提取所有点的x和y坐标
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # 计算边界框的坐标
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    return [x_min, y_min, x_max, y_max]

class ImageOCR:
    def __init__(self, score_thresh=0.8):
        self.ocr = PaddleOCR(lang="ch", use_gpu=True)
        self.score_thresh = score_thresh

    def predict(self, image: np.ndarray):
        ocr_res = self.ocr.ocr(image, det=True, rec=True, cls=True)[0]
        res = {"bboxs":[], "texts":[], "score":[]}
        for item in ocr_res:
            score = item[1][1]
            if score < self.score_thresh:
                continue
            res["score"].append(score)
            text = item[1][0]
            res["texts"].append(text)
            points = item[0]
            bbox = points_to_bbox(points)
            res["bboxs"].append(bbox)
        return res

if __name__ == '__main__':
    ocr = ImageOCR()
    from PIL import Image
    image = np.array(Image.open(r'E:\xingchen\memory_data\images\2025-05-06_21-20-00.856.png').convert('RGB'))
    result = ocr.predict(image)

    from image_module.tools.color_tools import apply_color_mapping_to_bboxs
    overlap_img = apply_color_mapping_to_bboxs(image, result["bboxs"])
    overlap_img.save('overlap.png')
