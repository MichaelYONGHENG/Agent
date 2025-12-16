"""
GUI ReAct Agent 主类
实现 感知-规划-执行 循环
"""
import time
import logging
from typing import Optional, Callable
from datetime import datetime

from core.config import Config, get_config
from core.perception.vision_model import VisionPerception
from core.planning.planner import ReActPlanner
from core.execution.action_executor import ActionExecutor
from core.execution.actions import ActionType
from core.memory.trajectory import Trajectory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GUIReactAgent:
    """
    GUI ReAct Agent
    
    基于纯视觉感知的GUI自动化Agent，实现：
    - 感知（Perception）：通过截图+多模态LLM理解界面
    - 规划（Planning）：基于ReAct范式决定下一步动作
    - 执行（Execution）：执行GUI操作（点击、滑动、键入等）
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        mode: str = "browser",
        headless: bool = False,
        save_trajectory: bool = True
    ):
        """
        初始化Agent
        
        Args:
            config: 配置对象，None则使用默认配置
            mode: 执行模式，"browser" 或 "desktop"
            headless: 浏览器是否无头模式
            save_trajectory: 是否保存执行轨迹
        """
        self.config = config or get_config()
        self.mode = mode
        self.save_trajectory = save_trajectory
        
        # 初始化各模块
        self.perception = VisionPerception(self.config)
        self.planner = ReActPlanner(self.config)
        self.executor = ActionExecutor(mode=mode, headless=headless)
        
        # 运行状态
        self.is_running = False
        self.current_task: Optional[str] = None
        self.trajectory: Optional[Trajectory] = None
        
        # 回调函数
        self.on_step_callback: Optional[Callable] = None
        
        logger.info(f"GUI ReAct Agent 初始化完成 (模式: {mode})")
    
    def run(
        self,
        task: str,
        start_url: Optional[str] = None,
        max_steps: Optional[int] = None,
        use_grounding: bool = False
    ) -> Trajectory:
        """
        执行任务主循环
        
        Args:
            task: 任务描述
            start_url: 起始URL（仅browser模式）
            max_steps: 最大步数，None则使用配置值
            use_grounding: 是否使用元素定位辅助规划
            
        Returns:
            执行轨迹
        """
        max_steps = max_steps or self.config.max_steps
        self.current_task = task
        self.is_running = True
        
        # 初始化轨迹记录
        self.trajectory = Trajectory(task) if self.save_trajectory else None
        
        logger.info("=" * 60)
        logger.info(f"🚀 开始任务: {task}")
        logger.info("=" * 60)
        
        try:
            # 启动执行器
            if self.mode == "browser":
                url = start_url or "https://www.google.com"
                self.executor.start(url)
                logger.info(f"浏览器已启动: {url}")
            
            # 主循环
            step = 0
            while step < max_steps and self.is_running:
                step += 1
                
                logger.info(f"\n{'='*50}")
                logger.info(f"📍 Step {step}/{max_steps}")
                logger.info(f"{'='*50}")
                
                # 执行一步
                should_stop = self._execute_step(step, use_grounding)
                
                if should_stop:
                    logger.info("✅ 任务完成或Agent主动停止")
                    break
                
                # 等待页面响应
                time.sleep(self.config.screenshot_interval)
            
            # 检查是否因为步数限制而停止
            if step >= max_steps:
                logger.warning(f"⚠️ 达到最大步数限制 ({max_steps})")
                if self.trajectory:
                    self.trajectory.mark_failed("达到最大步数限制")
            elif self.trajectory:
                self.trajectory.mark_completed()
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ 用户中断执行")
            if self.trajectory:
                self.trajectory.mark_failed("用户中断")
        except Exception as e:
            logger.error(f"❌ 执行出错: {e}")
            if self.trajectory:
                self.trajectory.mark_failed(str(e))
            raise
        finally:
            self._cleanup()
        
        # 打印摘要
        if self.trajectory:
            self.trajectory.print_summary()
        
        return self.trajectory
    
    def _execute_step(self, step: int, use_grounding: bool = False) -> bool:
        """
        执行单个步骤
        
        Args:
            step: 当前步骤编号
            use_grounding: 是否使用元素定位
            
        Returns:
            bool: 是否应该停止
        """
        # 1. 感知阶段
        logger.info("👁️  感知中...")
        screenshot = self.executor.take_screenshot()
        perception_result = self.perception.perceive(screenshot, self.current_task)
        
        logger.info(f"   场景: {perception_result.get('scene_description', 'N/A')}")
        logger.info(f"   状态: {perception_result.get('current_state', 'N/A')}")
        
        # 检查任务是否已完成
        if perception_result.get("is_task_complete", False):
            logger.info("🎉 感知模块判断任务已完成")
            return True
        
        # 2. 规划阶段
        logger.info("\n🧠 规划中...")
        history = self.trajectory.get_history(last_n=5) if self.trajectory else []
        
        if use_grounding:
            action = self.planner.plan_with_grounding(
                task=self.current_task,
                screenshot=screenshot,
                perception_result=perception_result,
                history=history,
                element_locator=self.perception.locate_element
            )
        else:
            action = self.planner.plan_next_action(
                task=self.current_task,
                screenshot=screenshot,
                perception_result=perception_result,
                history=history
            )
        
        logger.info(f"   决策: {action}")
        if action.thought:
            logger.info(f"   推理: {action.thought}")
        
        # 3. 执行阶段
        logger.info("\n⚡ 执行中...")
        success = self.executor.execute(action)
        
        if not success:
            logger.warning("   ❌ 动作执行失败")
        else:
            logger.info("   ✓ 动作执行成功")
        
        # 记录轨迹
        if self.trajectory:
            self.trajectory.add_step(
                action=action.to_dict(),
                thought=action.thought,
                perception=perception_result,
                success=success,
                screenshot=screenshot
            )
        
        # 调用回调
        if self.on_step_callback:
            self.on_step_callback(step, action, perception_result, success)
        
        # 检查是否停止
        if action.action_type == ActionType.STOP:
            return True
        
        return False
    
    def stop(self):
        """停止当前任务"""
        logger.info("🛑 收到停止信号")
        self.is_running = False
    
    def _cleanup(self):
        """清理资源"""
        logger.info("\n🧹 清理资源...")
        self.executor.close()
        self.planner.reset()
        self.is_running = False
        self.current_task = None
    
    def set_step_callback(self, callback: Callable):
        """
        设置每步执行后的回调函数
        
        Args:
            callback: 回调函数，签名为 (step, action, perception, success)
        """
        self.on_step_callback = callback
    
    def get_status(self) -> dict:
        """获取Agent当前状态"""
        return {
            "is_running": self.is_running,
            "current_task": self.current_task,
            "mode": self.mode,
            "steps_completed": len(self.trajectory) if self.trajectory else 0
        }


def create_agent(
    mode: str = "browser",
    headless: bool = False,
    **config_overrides
) -> GUIReactAgent:
    """
    工厂函数：创建Agent实例
    
    Args:
        mode: 执行模式
        headless: 是否无头模式
        **config_overrides: 配置覆盖参数
        
    Returns:
        GUIReactAgent 实例
    """
    config = Config(**config_overrides) if config_overrides else get_config()
    return GUIReactAgent(config=config, mode=mode, headless=headless)

