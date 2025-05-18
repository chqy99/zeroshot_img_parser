import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import math

def generate_even_palette(n):
    """
    根据颜色均分的方法生成一个长度为 n 的颜色表。

    Args:
        n (int): 颜色表的长度。

    Returns:
        list: 颜色表，每个颜色为 (R, G, B) 格式。
    """
    cm = plt.get_cmap('hsv')
    colors = [cm(i / n) for i in range(n)]
    palette = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]
    return palette

def generate_random_palette(n, seed=42):
    """
    根据随机方法生成一个长度为 n 的颜色表。

    Args:
        n (int): 颜色表的长度。
        seed (int, optional): 随机种子，用于保证结果可复现。默认为 42。

    Returns:
        list: 颜色表，每个颜色为 (R, G, B) 格式。
    """
    rng = np.random.default_rng(seed=seed)
    palette = rng.integers(0, 255, size=(n, 3))
    return palette.tolist()

def apply_color_mapping(gray_image: np.array, num_categories, palette=None, alpha=150):
    """
    将灰度图像转换为彩色图像，每个灰度值（类别）分配不同的颜色。

    参数:
        gray_image (np.ndarray): 输入灰度图像，是一个二维 numpy 数组，形状为 (height, width)，值为 0 或 1 或其他类别索引。
        num_categories (int): 类别数量（不包括背景）。
        palette (list, optional): 颜色表，如果为 None，则自动生成。默认为 None。
        alpha (int, optional): 透明度值，范围为 0-255。默认为 150。

    返回:
        PIL.Image.Image: 彩色图像，包含 RGBA 通道。
        如果 num_categories 为 0 或 gray_image 为 None，则返回 None。
    """
    if num_categories == 0 or gray_image is None:
        return None

    # 颜色表
    if palette is None:
        _palette = generate_even_palette(math.ceil(num_categories / 8) * 8)
    else:
        _palette = palette

    # 初始化彩色图像，添加 alpha 通道
    colored_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 4), dtype=np.uint8)

    # 应用颜色到每个类别
    for idx in range(1, num_categories + 1):  # 从1开始，因为0是背景
        if idx < len(_palette):
            color = _palette[idx - 1] + (alpha,)  # 添加透明度
            colored_image[gray_image == idx] = color

    # 创建图像
    colored_image_img = Image.fromarray(colored_image, mode="RGBA")

    return colored_image_img

def apply_color_mapping_to_masks(masks: list[np.ndarray], palette=None, alpha=150):
    """
    将多个二值掩码转换为彩色图像，每个掩码分配不同的颜色，带有透明度通道。

    参数:
        masks (List[np.ndarray]): 输入掩码列表，每个掩码是一个 bool 型二维 numpy 数组，形状为 (height, width)。
        palette (list, optional): 颜色表，如果为 None，则自动生成。默认为 None。
        alpha (int, optional): 透明度值，范围为 0-255。默认为 150。

    返回:
        np.ndarray: 彩色图像，包含 RGBA 通道。
        如果 masks 为空，则返回 None。
    """
    if not masks or len(masks) == 0:
        return None

    # 获取掩码的形状
    height, width = masks[0].shape[:2]

    # 如果未提供颜色表，则自动生成
    if palette is None:
        num_masks = len(masks)
        _palette = generate_even_palette(num_masks)
    else:
        _palette = palette

    # 初始化彩色图像，包含 RGBA 通道
    colored_image = np.zeros((height, width, 4), dtype=np.uint8)

    # 为每个掩码分配颜色并应用
    for i, mask in enumerate(masks):
        if i < len(_palette):
            color = _palette[i] + (alpha,)  # 添加透明度通道
            # 将颜色应用到掩码区域
            colored_image[mask] = color

    return colored_image

def composite_overlap_image(base_img: np.ndarray, overlay_img: np.ndarray) -> Image.Image:
    """
    将叠加图像叠加到基础图像上，并将结果保存到指定路径。

    参数:
        base_img (np.ndarray): 基础图像，形状为 (height, width, channels)，可以是 RGB 或 RGBA。
        overlay_img (np.ndarray): 需要叠加的图像，形状为 (height, width, 4)，包含 RGBA 通道。
        output_path (str): 合成后图像的保存路径。

    返回:
        None
    """
    # 将 NumPy 数组转换为 PIL Image 对象
    base_pil = Image.fromarray(base_img)
    overlay_pil = Image.fromarray(overlay_img)

    # 确保基础图像是 RGBA 模式
    if base_pil.mode != 'RGBA':
        base_pil = base_pil.convert('RGBA')

    # 确保叠加图像是 RGBA 模式
    if overlay_pil.mode != 'RGBA':
        overlay_pil = overlay_pil.convert('RGBA')

    # 将叠加图像叠加到基础图像上
    composite = Image.alpha_composite(base_pil, overlay_pil)
    return composite

def apply_color_mapping_to_bboxs(image: np.ndarray, bboxs: list[list], color=(255,0,0), is_composite=True, alpha=150):
    """
    将边界框列表转换为彩色图像，每个边界框分配不同的颜色，并带有透明度通道。

    参数:
        image (np.ndarray): 输入图像，是一个三维 numpy 数组，形状为 (height, width, channels)。
        bboxs (List[list]): 输入边界框列表，每个边界框是一个列表，格式为 [x_min, y_min, x_max, y_max]。
        is_composite (bool): 是否将边界框图像与原图像合成。默认为 True。
        palette (list, optional): 颜色表，如果为 None，则自动生成。默认为 None。
        alpha (int, optional): 透明度值，范围为 0-255。默认为 150。

    返回:
        Image: 彩色图像，包含 RGBA 通道。
    """
    # 将 NumPy 数组转换为 PIL 图像
    pil_image = Image.fromarray(image)

    # 创建一个与图像大小相同的透明图层，用于绘制边界框
    overlay = Image.new('RGBA', pil_image.size)
    draw = ImageDraw.Draw(overlay)

    # 为每个边界框分配颜色并绘制
    for bbox in bboxs:
        # 绘制边界框到 overlay 图层
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color + (alpha,), width=2)

    # 如果需要合成原图像，则将 overlay 图层与原图像混合
    if is_composite:
        # 将原图像转换为 RGBA 模式
        vis_image = pil_image.convert('RGBA')
        # 混合 overlay 和原图像
        vis_image = Image.alpha_composite(vis_image, overlay)
        return vis_image
    else:
        # 如果不需要合成原图像，则直接返回 overlay 图层
        return overlay
