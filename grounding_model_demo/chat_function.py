from openai import OpenAI
import numpy as np
import cv2
import base64
from config import Config

config = Config()

def encode_image_to_base64(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
        
    # 将NumPy数组编码为PNG格式的图片
    success, buffer = cv2.imencode('.png', image)
    if not success:
        return None
        
    # 将图片转换为base64字符串
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def grounding(text, image):
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    base64_image = encode_image_to_base64(image)
    response = client.chat.completions.create(
        model=config.grounding_model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=3000
    )
    import json
    import re

    content = response.choices[0].message.content

    # 检查是否包含```json块
    json_match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            if "bbox_2d" in data:
                return data["bbox_2d"]
            else:
                return data  # 没有bbox_2d就返回整个对象
        except Exception as e:
            # 如果解析失败，返回原内容
            return content
    else:
        # 不是json格式，直接返回原内容
        try:
            data = json.loads(content)
            if "bbox_2d" in data:
                return data["bbox_2d"]
            else:
                return data  # 没有bbox_2d就返回整个对象
        except Exception as e:
            # 如果解析失败，返回原内容
            return content
