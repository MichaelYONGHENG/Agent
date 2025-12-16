"""
动作空间定义模块
定义所有可用的GUI操作动作
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any


class ActionType(Enum):
    """动作类型枚举"""
    CLICK_LEFT = "click_left"       # 左键点击
    CLICK_RIGHT = "click_right"     # 右键点击
    SCROLL_UP = "scroll_up"         # 向上滑动
    SCROLL_DOWN = "scroll_down"     # 向下滑动
    TYPE = "type"                   # 键入文本
    WAIT = "wait"                   # 等待
    STOP = "stop"                   # 停止任务


@dataclass
class Action:
    """
    动作数据类
    表示Agent可执行的单个动作
    """
    action_type: ActionType
    coordinates: Optional[Tuple[int, int]] = None   # 点击/键入位置 (x, y)
    text: Optional[str] = None                       # 键入的文本
    scroll_amount: Optional[int] = None              # 滑动量（像素）
    wait_time: Optional[float] = None                # 等待时间（秒）
    thought: Optional[str] = None                    # 执行此动作的推理
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __repr__(self):
        parts = [f"Action({self.action_type.value}"]
        if self.coordinates:
            parts.append(f"coords={self.coordinates}")
        if self.text:
            parts.append(f"text='{self.text[:20]}..'" if len(self.text or '') > 20 else f"text='{self.text}'")
        if self.scroll_amount:
            parts.append(f"scroll={self.scroll_amount}")
        if self.wait_time:
            parts.append(f"wait={self.wait_time}s")
        return ", ".join(parts) + ")"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "action_type": self.action_type.value,
            "coordinates": self.coordinates,
            "text": self.text,
            "scroll_amount": self.scroll_amount,
            "wait_time": self.wait_time,
            "thought": self.thought,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """从字典创建Action"""
        return cls(
            action_type=ActionType(data["action_type"]),
            coordinates=tuple(data["coordinates"]) if data.get("coordinates") else None,
            text=data.get("text"),
            scroll_amount=data.get("scroll_amount"),
            wait_time=data.get("wait_time"),
            thought=data.get("thought"),
            metadata=data.get("metadata", {})
        )


class ActionSpace:
    """
    动作空间管理类
    提供便捷的动作创建方法
    """
    
    # 动作空间描述（用于Prompt）
    ACTION_SPACE_DESCRIPTION = """
## 可用动作空间

1. **click_left(x, y)** - 在坐标(x, y)处执行左键点击
   - 用于：点击按钮、链接、输入框等

2. **click_right(x, y)** - 在坐标(x, y)处执行右键点击  
   - 用于：打开右键菜单

3. **scroll_up(amount)** - 向上滚动页面
   - amount: 滚动像素数，默认300
   - 用于：查看页面上方内容

4. **scroll_down(amount)** - 向下滚动页面
   - amount: 滚动像素数，默认300
   - 用于：查看页面下方内容

5. **type(text, x, y)** - 在指定位置键入文本
   - text: 要输入的文本内容
   - x, y: 可选，先点击该位置再输入
   - 用于：填写表单、搜索框输入等

6. **wait(seconds)** - 等待指定时间
   - seconds: 等待秒数
   - 用于：等待页面加载、动画完成等

7. **stop()** - 任务完成，停止执行
   - 用于：任务目标已达成时调用
"""
    
    @staticmethod
    def create_click_left(x: int, y: int, thought: str = None) -> Action:
        """创建左键点击动作"""
        return Action(
            ActionType.CLICK_LEFT, 
            coordinates=(x, y),
            thought=thought
        )
    
    @staticmethod
    def create_click_right(x: int, y: int, thought: str = None) -> Action:
        """创建右键点击动作"""
        return Action(
            ActionType.CLICK_RIGHT, 
            coordinates=(x, y),
            thought=thought
        )
    
    @staticmethod
    def create_scroll_up(amount: int = 300, thought: str = None) -> Action:
        """创建向上滑动动作"""
        return Action(
            ActionType.SCROLL_UP, 
            scroll_amount=amount,
            thought=thought
        )
    
    @staticmethod
    def create_scroll_down(amount: int = 300, thought: str = None) -> Action:
        """创建向下滑动动作"""
        return Action(
            ActionType.SCROLL_DOWN, 
            scroll_amount=amount,
            thought=thought
        )
    
    @staticmethod
    def create_type(text: str, coordinates: Optional[Tuple[int, int]] = None, 
                    thought: str = None) -> Action:
        """创建键入文本动作"""
        return Action(
            ActionType.TYPE, 
            text=text, 
            coordinates=coordinates,
            thought=thought
        )
    
    @staticmethod
    def create_wait(seconds: float = 1.0, thought: str = None) -> Action:
        """创建等待动作"""
        return Action(
            ActionType.WAIT, 
            wait_time=seconds,
            thought=thought
        )
    
    @staticmethod
    def create_stop(thought: str = None) -> Action:
        """创建停止动作"""
        return Action(
            ActionType.STOP,
            thought=thought
        )
    
    @classmethod
    def from_llm_response(cls, response: Dict[str, Any]) -> Action:
        """
        从LLM响应创建Action
        
        Expected format:
        {
            "thought": "推理过程",
            "action": "动作名称",
            "parameters": {"参数": "值"}
        }
        """
        action_name = response.get("action", "").lower()
        params = response.get("parameters", {})
        thought = response.get("thought")
        
        if action_name == "click_left":
            return cls.create_click_left(
                x=int(params.get("x", 0)),
                y=int(params.get("y", 0)),
                thought=thought
            )
        elif action_name == "click_right":
            return cls.create_click_right(
                x=int(params.get("x", 0)),
                y=int(params.get("y", 0)),
                thought=thought
            )
        elif action_name == "scroll_up":
            return cls.create_scroll_up(
                amount=int(params.get("amount", 300)),
                thought=thought
            )
        elif action_name == "scroll_down":
            return cls.create_scroll_down(
                amount=int(params.get("amount", 300)),
                thought=thought
            )
        elif action_name == "type":
            coords = None
            if "x" in params and "y" in params:
                coords = (int(params["x"]), int(params["y"]))
            return cls.create_type(
                text=str(params.get("text", "")),
                coordinates=coords,
                thought=thought
            )
        elif action_name == "wait":
            return cls.create_wait(
                seconds=float(params.get("seconds", 1.0)),
                thought=thought
            )
        elif action_name == "stop":
            return cls.create_stop(thought=thought)
        else:
            raise ValueError(f"未知动作类型: {action_name}")

