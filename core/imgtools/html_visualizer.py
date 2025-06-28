import html
from typing import List
from core.imgdata.image_data import ImageParseResult, ImageParseItem


def generate_html_for_result(
    result: ImageParseResult, show_fields: List[str] = None
) -> str:
    """
    生成一个 HTML 页面，展示 ImageParseResult 各 ImageParseItem 的内容。
    show_fields 控制显示哪些字段，默认显示主要文本字段和图片。
    """
    if show_fields is None:
        show_fields = [
            "source_module",
            "score",
            "type",
            "label",
            "text",
            "bbox_image",
            "mask_image",
        ]

    # 表头HTML
    headers_html = "".join(f"<th>{html.escape(field)}</th>" for field in show_fields)

    rows_html = ""
    for item in result.items:
        # 用 to_dict 转为 dict，filter 触发 base64 图像生成
        item_dict = item.to_dict(filter=["bbox_image", "mask_image"])

        cells = []
        for field in show_fields:
            val = item_dict.get(field, None)
            if val is None:
                cells.append("<td></td>")
            elif field in ("bbox_image", "mask_image"):
                # 图片字段，val 是base64字符串，生成 img标签
                cells.append(
                    f'<td><img src="data:image/png;base64,{val}" style="max-width:120px;"/></td>'
                )
            elif isinstance(val, list):
                # 数组字段转成逗号字符串，防止过长截断
                short_val = ", ".join(map(str, val))
                if len(short_val) > 50:
                    short_val = short_val[:47] + "..."
                cells.append(f"<td>{html.escape(short_val)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(val))}</td>")

        rows_html += "<tr>" + "".join(cells) + "</tr>"

    html_str = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>ImageParseResult Visualization</title>
    <style>
    body {{ font-family: Arial, sans-serif; padding: 10px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; vertical-align: middle; }}
    th {{ background-color: #f2f2f2; }}
    img {{ max-width: 120px; height: auto; }}
    </style>
    </head>
    <body>
    <h1>ImageParseResult Visualization</h1>
    <table>
    <thead><tr>{headers_html}</tr></thead>
    <tbody>
    {rows_html}
    </tbody>
    </table>
    </body>
    </html>
    """
    return html_str
