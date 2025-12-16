"""
规划器模块
基于ReAct（Reasoning + Acting）范式的任务规划
"""
import json
import re
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from core.config import get_config
from core.llm.client import VisionChatClient
from core.execution.actions import Action, ActionSpace

logger = logging.getLogger(__name__)


class ReActPlanner:
    """
    ReAct规划器
    实现 Thought -> Action -> Observation 循环
    """
    
    SYSTEM_PROMPT = """你是一个专业的GUI自动化Agent规划器。
你的任务是根据当前界面和任务目标，决定下一步应该执行什么动作。

## 规划原则

1. **目标导向**：每个动作都应该朝着完成任务的方向前进
2. **最小动作**：优先选择最简单直接的动作
3. **状态感知**：根据当前界面状态做出合理判断
4. **错误处理**：如果检测到错误或异常，尝试恢复

## 动作空间

你可以使用以下动作：

1. `click_left(x, y)` - 在坐标(x, y)处左键点击
2. `click_right(x, y)` - 在坐标(x, y)处右键点击  
3. `scroll_up(amount)` - 向上滚动，amount默认300
4. `scroll_down(amount)` - 向下滚动，amount默认300
5. `type(text, x, y)` - 键入文本，可选先点击(x,y)位置
6. `wait(seconds)` - 等待指定秒数
7. `stop()` - 任务完成，停止执行

## 输出格式

请使用以下JSON格式输出：

```json
{
    "thought": "你的推理过程（分析当前状态，为什么选择这个动作）",
    "action": "动作名称",
    "parameters": {
        "参数名": "参数值"
    }
}
```

## 注意事项

- 坐标(x, y)是像素坐标，相对于截图左上角
- 先仔细观察截图，找到目标元素的位置
- 如果需要输入文本，确保先点击输入框
- 任务完成时调用 stop() 动作
"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.planner_client = VisionChatClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            system_prompt=self.SYSTEM_PROMPT
        )
    
    def plan_next_action(
        self,
        task: str,
        screenshot: np.ndarray,
        perception_result: Dict[str, Any],
        history: List[Dict],
        max_history: int = 5
    ) -> Action:
        """
        规划下一个动作
        
        Args:
            task: 任务描述
            screenshot: 当前界面截图
            perception_result: 视觉感知结果
            history: 历史轨迹
            max_history: 保留的最大历史步数
            
        Returns:
            下一个要执行的动作
        """
        # 获取图像尺寸
        height, width = screenshot.shape[:2]
        
        # 构建历史轨迹描述
        trajectory_text = self._format_trajectory(history[-max_history:])
        
        # 构建规划提示词
        prompt = f"""
## 当前任务
{task}

## 界面信息
- 截图尺寸: {width} x {height} 像素
- 场景描述: {perception_result.get('scene_description', '无')}
- 页面类型: {perception_result.get('page_type', '未知')}
- 可交互元素: {perception_result.get('interactive_elements', [])}
- 当前状态: {perception_result.get('current_state', '未知')}
- 建议动作: {perception_result.get('suggestions', [])}

## 历史轨迹
{trajectory_text}

## 你的任务
分析当前截图，结合任务目标和历史轨迹，决定下一步动作。

请返回JSON格式的决策：
{{
    "thought": "你的推理过程",
    "action": "动作名称",
    "parameters": {{参数}}
}}
"""
        
        try:
            # 发送带图像的请求
            self.planner_client.add_user_message_with_image(prompt, screenshot)
            
            response = self.planner_client.chat_completion(
                model=self.config.reasoning_model,
                temperature=0.4,
                max_tokens=1000
            )
            
            # 解析响应
            decision = self._parse_json_response(response)
            
            # 转换为Action对象
            action = ActionSpace.from_llm_response(decision)
            
            logger.info(f"规划动作: {action}")
            if action.thought:
                logger.info(f"推理: {action.thought}")
            
            return action
            
        except Exception as e:
            logger.error(f"规划失败: {e}")
            # 失败时返回等待动作
            return ActionSpace.create_wait(
                seconds=2.0,
                thought=f"规划失败，等待重试: {str(e)}"
            )
    
    def plan_with_grounding(
        self,
        task: str,
        screenshot: np.ndarray,
        perception_result: Dict[str, Any],
        history: List[Dict],
        element_locator: callable = None
    ) -> Action:
        """
        带元素定位的规划
        先决定要操作什么元素，再定位该元素
        
        Args:
            task: 任务描述
            screenshot: 当前界面截图
            perception_result: 视觉感知结果
            history: 历史轨迹
            element_locator: 元素定位函数
            
        Returns:
            下一个要执行的动作
        """
        height, width = screenshot.shape[:2]
        trajectory_text = self._format_trajectory(history[-5:])
        
        # 第一步：决定操作意图
        intent_prompt = f"""
## 当前任务
{task}

## 界面信息
- 场景: {perception_result.get('scene_description', '无')}
- 可交互元素: {perception_result.get('interactive_elements', [])}
- 当前状态: {perception_result.get('current_state', '未知')}

## 历史轨迹
{trajectory_text}

## 你的任务
决定下一步应该做什么。

返回JSON格式：
{{
    "thought": "推理过程",
    "intent": "操作意图（如：点击搜索按钮、输入查询词等）",
    "target_element": "目标元素描述（如：搜索按钮、输入框等）",
    "action_type": "动作类型（click_left/type/scroll_up/scroll_down/wait/stop）",
    "text_to_type": "如果是type动作，要输入的文本"
}}
"""
        
        try:
            response = self.planner_client.chat_completion_with_image(
                model=self.config.reasoning_model,
                text=intent_prompt,
                image=screenshot,
                temperature=0.4,
                max_tokens=800
            )
            
            intent = self._parse_json_response(response)
            action_type = intent.get("action_type", "wait")
            
            # 处理不需要坐标的动作
            if action_type == "stop":
                return ActionSpace.create_stop(thought=intent.get("thought"))
            elif action_type == "wait":
                return ActionSpace.create_wait(seconds=2.0, thought=intent.get("thought"))
            elif action_type == "scroll_up":
                return ActionSpace.create_scroll_up(thought=intent.get("thought"))
            elif action_type == "scroll_down":
                return ActionSpace.create_scroll_down(thought=intent.get("thought"))
            
            # 需要坐标的动作：定位元素
            target_element = intent.get("target_element", "")
            
            if element_locator and target_element:
                coords = element_locator(screenshot, target_element)
                if coords:
                    x, y = coords
                    if action_type == "click_left":
                        return ActionSpace.create_click_left(x, y, thought=intent.get("thought"))
                    elif action_type == "type":
                        text = intent.get("text_to_type", "")
                        return ActionSpace.create_type(text, (x, y), thought=intent.get("thought"))
            
            # 定位失败，回退到普通规划
            logger.warning(f"元素定位失败: {target_element}，使用普通规划")
            return self.plan_next_action(task, screenshot, perception_result, history)
            
        except Exception as e:
            logger.error(f"规划失败: {e}")
            return ActionSpace.create_wait(seconds=2.0, thought=f"规划失败: {str(e)}")
    
    def _format_trajectory(self, history: List[Dict]) -> str:
        """格式化历史轨迹"""
        if not history:
            return "（无历史记录，这是第一步）"
        
        lines = []
        for i, step in enumerate(history):
            step_num = step.get("step", i + 1)
            action = step.get("action", "N/A")
            thought = step.get("thought", "")
            success = "✓" if step.get("success", True) else "✗"
            
            lines.append(f"Step {step_num} {success}: {action}")
            if thought:
                lines.append(f"  └ 推理: {thought[:100]}...")
        
        return "\n".join(lines)
    
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
        return {
            "thought": "解析失败",
            "action": "wait",
            "parameters": {"seconds": 2}
        }
    
    def reset(self):
        """重置规划器状态"""
        self.planner_client.clear_history(keep_system=True)

