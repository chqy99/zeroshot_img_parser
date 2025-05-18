import numpy as np

def mask_to_bbox(mask):
        """
        将二值分割掩码转换为边界框坐标（左上角和右下角坐标）。

        参数:
        mask -- 二维的分割掩码，是一个 numpy 数组，形状为 (height, width)，值为 0 或 1。

        返回:
        返回一个列表，包含边界框的左上角和右下角坐标，格式为 [x1, y1, x2, y2]。
        如果掩码全为 0，则返回 None。
        """
        # 找到掩码中非零元素的行和列索引
        rows = np.where(np.any(mask, axis=1))[0]  # 行索引
        cols = np.where(np.any(mask, axis=0))[0]  # 列索引

        if len(rows) == 0 or len(cols) == 0:
            # 如果掩码全为 0，返回 None
            return None

        # 获取边界框的坐标
        y1 = rows[0]
        y2 = rows[-1]
        x1 = cols[0]
        x2 = cols[-1]

        return [x1, y1, x2, y2]
