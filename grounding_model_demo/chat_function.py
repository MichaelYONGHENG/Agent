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
        max_tokens=300
    )
    return response.choices[0].message.content
