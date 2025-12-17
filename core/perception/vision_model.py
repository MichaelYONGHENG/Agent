"""
视觉感知模块
基于多模态LLM的界面理解和元素定位
使用阿里云 Qwen3-VL-Plus 模型
"""
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.config import get_config
from core.llm.client import VisionChatClient

logger = logging.getLogger(__name__)


class VisionPerception:
    """
    视觉感知类
    负责理解当前界面状态和定位UI元素
    使用阿里云 Qwen3-VL-Plus 模型进行视觉理解
    """
    
    # 感知系统提示词
    PERCEPTION_SYSTEM_PROMPT = """你是一个GUI自动化Agent的视觉感知模块。
你的任务是分析界面截图，理解当前状态，并提供准确的场景描述。

你需要：
1. 准确描述当前界面的整体布局和内容
2. 识别所有可交互的元素（按钮、链接、输入框等）
3. 判断任务的当前进展状态
4. 如果任务已完成，明确指出

请始终使用中文回复，并保持描述简洁准确。"""

    # 元素定位系统提示词
    GROUNDING_SYSTEM_PROMPT = """你是一个UI元素定位专家。
给定一个界面截图和目标元素描述，你需要返回该元素的精确坐标。

坐标格式要求：
- 返回点击位置的中心坐标 (x, y)
- 坐标使用像素值，相对于图像左上角
- 如果找不到目标元素，返回 null

请只返回JSON格式的结果。"""
    
    def __init__(self, config=None):
        """
        初始化视觉感知模块
        
        Args:
            config: 配置对象，None则使用默认配置
        """
        self.config = config or get_config()
        
        # 使用阿里云配置创建视觉客户端
        self.vision_client = VisionChatClient(
            api_key=self.config.perception_api_key,
            base_url=self.config.perception_base_url
        )
        self.grounding_client = VisionChatClient(
            api_key=self.config.perception_api_key,
            base_url=self.config.perception_base_url
        )
        
        logger.info(f"视觉感知模块初始化完成，使用模型: {self.config.vision_model}")
    
    def perceive(self, screenshot: np.ndarray, task_description: str) -> Dict[str, Any]:
        """
        感知当前界面状态
        
        Args:
            screenshot: 当前界面截图 (numpy array, BGR格式)
            task_description: 任务描述
            
        Returns:
            感知结果字典，包含：
            - scene_description: 场景描述
            - interactive_elements: 可交互元素列表
            - current_state: 当前任务状态
            - is_task_complete: 任务是否完成
            - suggestions: 建议的下一步动作
        """
        prompt = f"""
分析这个界面截图。当前正在执行的任务是：{task_description}

请返回JSON格式的分析结果：
{{
    "scene_description": "当前页面的整体描述（一句话）",
    "page_type": "页面类型（如：搜索结果页、登录页、表单页等）",
    "interactive_elements": ["列出主要可交互元素"],
    "current_state": "任务当前进展状态描述",
    "is_task_complete": false,
    "error_detected": null,
    "suggestions": ["建议的下一步动作"]
}}

注意：只返回JSON，不要有其他文字。
"""
        
        try:
            response = self.vision_client.chat_completion_with_image(
                model=self.config.vision_model,
                text=prompt,
                image=screenshot,
                temperature=0.3,
                max_tokens=1500
            )
            
            # 解析JSON响应
            result = self._parse_json_response(response)
            logger.info(f"感知结果: {result.get('scene_description', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"感知失败: {e}")
            return {
                "scene_description": "感知失败",
                "interactive_elements": [],
                "current_state": "未知",
                "is_task_complete": False,
                "error_detected": str(e),
                "suggestions": ["重试感知"]
            }
    
    def locate_element(self, screenshot: np.ndarray, element_description: str) -> Optional[Tuple[int, int]]:
        """
        定位UI元素
        
        Args:
            screenshot: 界面截图
            element_description: 元素描述（如 "登录按钮"、"搜索框"）
            
        Returns:
            元素中心坐标 (x, y)，如果找不到返回 None
        """
        # 获取图像尺寸
        height, width = screenshot.shape[:2]
        
        prompt = f"""
在这个界面截图中定位以下元素："{element_description}"

请返回该元素的点击坐标（元素中心位置）。
图像尺寸：宽度={width}像素，高度={height}像素

返回JSON格式：
{{
    "found": true,
    "x": 像素坐标X,
    "y": 像素坐标Y,
    "confidence": "high/medium/low",
    "element_type": "元素类型"
}}

如果找不到元素，返回：
{{"found": false, "reason": "找不到的原因"}}

只返回JSON，不要有其他文字。
"""
        
        try:
            response = self.grounding_client.chat_completion_with_image(
                model=self.config.grounding_model,
                text=prompt,
                image=screenshot,
                temperature=0.2,
                max_tokens=500
            )
            
            result = self._parse_json_response(response)
            
            if result.get("found", False):
                x = int(result.get("x", 0))
                y = int(result.get("y", 0))
                
                # 验证坐标在有效范围内
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                
                logger.info(f"定位元素 '{element_description}': ({x}, {y})")
                return (x, y)
            else:
                logger.warning(f"未找到元素 '{element_description}': {result.get('reason', '未知原因')}")
                return None
                
        except Exception as e:
            logger.error(f"元素定位失败: {e}")
            return None
    
    def locate_element_bbox(self, screenshot: np.ndarray, element_description: str) -> Optional[List[int]]:
        """
        定位UI元素的边界框
        
        Args:
            screenshot: 界面截图
            element_description: 元素描述
            
        Returns:
            边界框坐标 [x1, y1, x2, y2]，如果找不到返回 None
        """
        height, width = screenshot.shape[:2]
        
        prompt = f"""
在这个界面截图中定位以下元素的边界框："{element_description}"

图像尺寸：宽度={width}像素，高度={height}像素

返回JSON格式：
{{
    "found": true,
    "bbox": [x1, y1, x2, y2],
    "confidence": "high/medium/low"
}}

如果找不到元素，返回：
{{"found": false, "reason": "找不到的原因"}}

只返回JSON，不要有其他文字。
"""
        
        try:
            response = self.grounding_client.chat_completion_with_image(
                model=self.config.grounding_model,
                text=prompt,
                image=screenshot,
                temperature=0.2,
                max_tokens=500
            )
            
            result = self._parse_json_response(response)
            
            if result.get("found", False):
                bbox = result.get("bbox", [])
                if len(bbox) == 4:
                    logger.info(f"定位元素边界框 '{element_description}': {bbox}")
                    return bbox
            
            return None
                
        except Exception as e:
            logger.error(f"边界框定位失败: {e}")
            return None
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析LLM的JSON响应"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试从markdown代码块中提取
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试找到JSON对象
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"无法解析JSON响应: {response[:200]}...")
        return {"raw_response": response}
