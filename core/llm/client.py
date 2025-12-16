"""
LLM 客户端模块
提供统一的 OpenAI API 调用接口，支持重试机制和上下文管理
"""
import os
import time
import functools
import logging
import base64
from typing import Optional, Dict, Any, List, Union
import numpy as np
import cv2
from openai import OpenAI

logger = logging.getLogger(__name__)


def retry_openai_call(max_retries: int = 3, base_delay: float = 1.0):
    """
    装饰器：为OpenAI API调用添加重试机制
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒），每次重试会指数增长
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"API调用失败，已达到最大重试次数 {max_retries}，最后一次错误: {e}")
                        raise e
                    else:
                        delay = base_delay * (2 ** (retry_count - 1))  # 指数增长延迟
                        logger.warning(f"API调用失败，第 {retry_count} 次重试，错误: {e}，等待 {delay} 秒后重试...")
                        time.sleep(delay)
            
            return None  # 这行代码实际不会执行到
        return wrapper
    return decorator


class ChatClient:
    """
    OpenAI客户端包装器，提供统一的配置和重试机制
    任务粒度的上下文管理，每个任务实例维护独立的对话历史
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 system_prompt: Optional[str] = None):
        # 从环境变量获取配置，如果没有提供参数的话
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        # 任务级别的上下文历史
        self.history_messages: List[Dict[str, Any]] = []
        
        # 如果提供了系统提示，添加到历史开头
        if system_prompt:
            self.history_messages.append({"role": "system", "content": system_prompt})
        
        if not self.api_key:
            raise ValueError("API密钥未配置。请在.env文件中设置OPENAI_API_KEY，或在初始化时提供api_key参数。")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def add_user_message(self, content: Union[str, List[Dict]]):
        """添加用户消息到历史记录"""
        self.history_messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        """添加助手消息到历史记录"""
        self.history_messages.append({"role": "assistant", "content": content})
    
    def add_system_message(self, content: str):
        """添加系统消息到历史记录"""
        self.history_messages.append({"role": "system", "content": content})
    
    def clear_history(self, keep_system: bool = True):
        """清空历史记录"""
        if keep_system:
            self.history_messages = [msg for msg in self.history_messages if msg["role"] == "system"]
        else:
            self.history_messages = []
    
    @retry_openai_call(max_retries=3, base_delay=1.0)
    def chat_completion(self, model: str,
                       temperature: float = 0.6, 
                       response_format: Optional[Dict[str, Any]] = None,
                       max_tokens: int = 2000) -> str:
        """
        带重试机制的聊天完成API调用
        """
        kwargs = {
            "model": model,
            "messages": self.history_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_format:
            kwargs["response_format"] = response_format
            
        response = self.client.chat.completions.create(**kwargs)
        response_content = response.choices[0].message.content
        
        self.add_assistant_message(response_content)
        
        return response_content
    
    def send_message(self, user_message: str, model: str, 
                     temperature: float = 0.6, 
                     response_format: Optional[Dict[str, Any]] = None,
                     max_tokens: int = 2000) -> str:
        """
        发送用户消息并获取回复，自动管理上下文历史
        """
        # 添加用户消息到历史
        self.add_user_message(user_message)
        
        # 获取模型回复（会自动添加到历史中）
        response = self.chat_completion(
            model=model, 
            temperature=temperature, 
            response_format=response_format,
            max_tokens=max_tokens
        )
        
        return response


class VisionChatClient(ChatClient):
    """
    支持视觉输入的聊天客户端
    继承自 ChatClient，添加图像处理功能
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 system_prompt: Optional[str] = None):
        super().__init__(api_key, base_url, system_prompt)
    
    @staticmethod
    def encode_image_to_base64(image: np.ndarray) -> str:
        """将 numpy 图像数组编码为 base64 字符串"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 确保是 BGR 格式（OpenCV 默认格式）
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("图像编码失败")
        
        return base64.b64encode(buffer).decode('utf-8')
    
    def add_user_message_with_image(self, text: str, image: np.ndarray):
        """添加带图像的用户消息"""
        base64_image = self.encode_image_to_base64(image)
        content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
        self.add_user_message(content)
    
    @retry_openai_call(max_retries=3, base_delay=1.0)
    def chat_completion_with_image(self, model: str, text: str, image: np.ndarray,
                                   temperature: float = 0.6,
                                   max_tokens: int = 2000) -> str:
        """
        发送带图像的消息并获取回复（不保存到历史，适合单次调用）
        """
        base64_image = self.encode_image_to_base64(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def send_message_with_image(self, text: str, image: np.ndarray, model: str,
                                temperature: float = 0.6,
                                max_tokens: int = 2000) -> str:
        """
        发送带图像的消息并获取回复，自动管理上下文历史
        """
        self.add_user_message_with_image(text, image)
        
        response = self.chat_completion(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response

