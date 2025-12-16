"""
轨迹记忆模块
记录Agent的执行历史，用于规划和调试
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """
    轨迹步骤数据类
    记录每一步的完整信息
    """
    step: int                                    # 步骤编号
    timestamp: str                               # 时间戳
    action: Dict[str, Any]                       # 执行的动作
    thought: Optional[str] = None                # 推理过程
    perception: Optional[Dict[str, Any]] = None  # 感知结果
    success: bool = True                         # 执行是否成功
    screenshot_path: Optional[str] = None        # 截图保存路径
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        """从字典创建"""
        return cls(**data)
    
    def summary(self) -> str:
        """生成步骤摘要"""
        status = "✓" if self.success else "✗"
        action_str = self.action.get("action_type", "unknown")
        return f"Step {self.step} {status}: {action_str}"


class Trajectory:
    """
    轨迹记录类
    管理整个任务的执行历史
    """
    
    def __init__(self, task: str, save_dir: Optional[str] = None):
        """
        初始化轨迹记录
        
        Args:
            task: 任务描述
            save_dir: 截图和轨迹保存目录
        """
        self.task = task
        self.steps: List[TrajectoryStep] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.status = "running"  # running, completed, failed
        
        # 设置保存目录
        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path(f"./trajectories/{timestamp}")
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir = self.save_dir / "screenshots"
        self.screenshots_dir.mkdir(exist_ok=True)
    
    def add_step(
        self,
        action: Dict[str, Any],
        thought: Optional[str] = None,
        perception: Optional[Dict[str, Any]] = None,
        success: bool = True,
        screenshot: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrajectoryStep:
        """
        添加一个步骤
        
        Args:
            action: 执行的动作（字典格式）
            thought: 推理过程
            perception: 感知结果
            success: 执行是否成功
            screenshot: 截图（numpy数组）
            metadata: 额外元数据
            
        Returns:
            创建的轨迹步骤
        """
        step_num = len(self.steps) + 1
        timestamp = datetime.now().isoformat()
        
        # 保存截图
        screenshot_path = None
        if screenshot is not None:
            screenshot_path = str(self.screenshots_dir / f"step_{step_num:03d}.png")
            cv2.imwrite(screenshot_path, screenshot)
        
        step = TrajectoryStep(
            step=step_num,
            timestamp=timestamp,
            action=action,
            thought=thought,
            perception=perception,
            success=success,
            screenshot_path=screenshot_path,
            metadata=metadata or {}
        )
        
        self.steps.append(step)
        logger.debug(f"记录步骤: {step.summary()}")
        
        return step
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取历史记录（用于传递给规划器）
        
        Args:
            last_n: 只返回最后N步，None表示返回全部
            
        Returns:
            历史记录列表
        """
        steps = self.steps[-last_n:] if last_n else self.steps
        
        return [
            {
                "step": s.step,
                "action": s.action.get("action_type", "unknown"),
                "thought": s.thought,
                "success": s.success,
                "perception_summary": s.perception.get("scene_description", "") if s.perception else ""
            }
            for s in steps
        ]
    
    def mark_completed(self):
        """标记任务完成"""
        self.status = "completed"
        self.end_time = datetime.now()
        self.save()
    
    def mark_failed(self, reason: str = ""):
        """标记任务失败"""
        self.status = "failed"
        self.end_time = datetime.now()
        if self.steps:
            self.steps[-1].metadata["failure_reason"] = reason
        self.save()
    
    def save(self, filename: str = "trajectory.json"):
        """保存轨迹到文件"""
        filepath = self.save_dir / filename
        
        data = {
            "task": self.task,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_steps": len(self.steps),
            "steps": [step.to_dict() for step in self.steps]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"轨迹已保存: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "Trajectory":
        """从文件加载轨迹"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        trajectory = cls(
            task=data["task"],
            save_dir=str(Path(filepath).parent)
        )
        trajectory.status = data["status"]
        trajectory.start_time = datetime.fromisoformat(data["start_time"])
        if data["end_time"]:
            trajectory.end_time = datetime.fromisoformat(data["end_time"])
        
        trajectory.steps = [
            TrajectoryStep.from_dict(step_data)
            for step_data in data["steps"]
        ]
        
        return trajectory
    
    def get_summary(self) -> Dict[str, Any]:
        """获取轨迹摘要"""
        duration = None
        if self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        success_count = sum(1 for s in self.steps if s.success)
        
        return {
            "task": self.task,
            "status": self.status,
            "total_steps": len(self.steps),
            "successful_steps": success_count,
            "failed_steps": len(self.steps) - success_count,
            "duration_seconds": duration,
            "save_dir": str(self.save_dir)
        }
    
    def print_summary(self):
        """打印轨迹摘要"""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("📊 任务轨迹摘要")
        print("=" * 60)
        print(f"任务: {summary['task']}")
        print(f"状态: {summary['status']}")
        print(f"总步数: {summary['total_steps']}")
        print(f"成功: {summary['successful_steps']}, 失败: {summary['failed_steps']}")
        if summary['duration_seconds']:
            print(f"耗时: {summary['duration_seconds']:.2f}秒")
        print(f"保存位置: {summary['save_dir']}")
        print("=" * 60)
        
        print("\n步骤详情:")
        for step in self.steps:
            print(f"  {step.summary()}")
            if step.thought:
                print(f"    └ {step.thought[:80]}...")
    
    def __len__(self):
        return len(self.steps)
    
    def __getitem__(self, index):
        return self.steps[index]

