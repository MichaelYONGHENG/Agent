import gradio as gr
import cv2
import numpy as np
import re
from chat_function import grounding

def draw_result_on_image(image, coordinates):
    result_image = np.array(image).copy()
    numbers = coordinates

    color = (255, 0, 0)  # BGR格式，红色
    thickness = 2

    image_shape = result_image.shape
    height, width = image_shape[:2]

    if len(numbers) == 2:
        # 如果是点坐标 (x, y)
        x, y = numbers
        x = int((x / 1000.0) * width)
        y = int((y / 1000.0) * height)
        # 画一个小圆点
        cv2.circle(result_image, (x, y), 5, color, -1)
    elif len(numbers) == 4:
        # 如果是框坐标 (x1, y1, x2, y2)
        x1, y1, x2, y2 = numbers
        # 将[0,1000]范围转换为实际图像坐标
        x1 = int((x1 / 1000.0) * width)
        y1 = int((y1 / 1000.0) * height)
        x2 = int((x2 / 1000.0) * width)
        y2 = int((y2 / 1000.0) * height)
        # 画矩形框
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
    return result_image


def process_grounding(image, text):
    if image is None:
        return None
    # 调用grounding函数获取坐标（这里假设函数已存在）
    coordinates = grounding(text, image)
    print(coordinates)
    # 在图像上绘制结果
    result_image = draw_result_on_image(image, coordinates)
    return result_image, coordinates

# 创建Gradio界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # 左侧：输入区域
            input_image = gr.Image(label="上传图片")
            text_input = gr.Textbox(label="输入目标元素名称")

        with gr.Column(scale=1):
            # 右侧：输出区域
            output_image = gr.Image(label="结果展示")
            output_coordinates = gr.Textbox(label="检测结果坐标")

    # 处理按钮
    submit_btn = gr.Button("开始检测")

    # 设置点击事件
    submit_btn.click(
        fn=process_grounding,
        inputs=[input_image, text_input],
        outputs=[output_image, output_coordinates]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()
