import matplotlib.pyplot as plt

def generate_palette(n):
    """
    生成一个长度为 n 的颜色表，采用均匀采样，然后转换为 RGB 格式。

    Args:
        n (int): 颜色表的长度。

    Returns:
        list: 颜色表，每个颜色为 (R, G, B) 格式。
    """
    # 使用 Matplotlib 生成均匀分布的颜色
    cm = plt.get_cmap('hsv')  # 使用 HSV 色彩映射
    colors = [cm(i / n) for i in range(n)]
    # 转换为 RGB 格式并缩放到 0-255
    palette = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]
    return palette
