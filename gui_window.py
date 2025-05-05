import pygetwindow as gw
import pyautogui
import mss
import time
import numpy as np


def get_windows_title(show_flag=True):
    windows = [title for title in gw.getAllTitles() if title]
    _count = {}

    for index, title in enumerate(windows):
        _count[title] = _count.get(title, -1) + 1
        if show_flag:
            print(f"index: {index}, title: {title}, title_idx: {_count[title]}")
    return windows


def get_window(window_title: str, title_idx: int):
    window = gw.getWindowsWithTitle(window_title)[title_idx]
    if window.isMinimized:
        window.restore()  # 恢复窗口
    window.activate()
    time.sleep(0.2)
    return window

def get_monitor(idx=1):
    with mss.mss() as sct:
        return sct.monitors[idx]

def get_screenshot(shot_rect: dict) -> np.array:
    with mss.mss() as sct:
        screenshot = sct.grab(shot_rect)

    screenshot_array = np.array(screenshot)
    # 转成 rgb 需要深拷贝，影响性能
    bgr_array = screenshot_array[:, :, :3]
    return bgr_array

def save_screenshot(shot_rect: dict, filepath: str):
    with mss.mss() as sct:
        screenshot = sct.grab(shot_rect)
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=filepath)

def parse_rect(*args, **kwargs):
    if len(args) == 1:
        # 如果传入的是一个列表或元组
        if isinstance(args[0], list) or isinstance(args[0], tuple):
            left, top, width, height = args[0]
        # 如果传入的是一个字典
        elif isinstance(args[0], dict):
            left = args[0].get("left")
            top = args[0].get("top")
            width = args[0].get("width")
            height = args[0].get("height")
    # 如果传入的是四个单独的参数
    elif len(args) == 4:
        left, top, width, height = args
    # 如果传入的是关键字参数
    elif (
        "left" in kwargs
        and "top" in kwargs
        and "width" in kwargs
        and "height" in kwargs
    ):
        left = kwargs["left"]
        top = kwargs["top"]
        width = kwargs["width"]
        height = kwargs["height"]
    else:
        raise ValueError("Invalid arguments for WindowRect initialization")

    return left, top, width, height


class GuiWindow:
    def __init__(self, window_title: str = "", title_idx: int = 0):
        self.window_name = window_title
        self.window = get_window(window_title, title_idx)
        self.left = self.window.left
        self.top = self.window.top
        self.width = self.window.width
        self.height = self.window.height

    def get_rect(self, format="dict"):
        if format == "tuple":
            return (self.left, self.top, self.width, self.height)
        elif format == "list":
            return [self.left, self.top, self.width, self.height]
        elif format == "dict":
            return {
                "left": self.left,
                "top": self.top,
                "width": self.width,
                "height": self.height,
            }
        else:
            return self.left, self.top, self.width, self.height

    def set_rect(self, *args, **kwargs):
        self.left, self.top, self.width, self.height = parse_rect(*args, **kwargs)

    def get_screenshot(self):
        rect = self.get_rect()
        return get_screenshot(rect)


if __name__ == "__main__":
    windows = get_windows_title()
