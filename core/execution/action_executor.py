"""
动作执行器模块
负责将动作转换为实际的GUI操作
支持 Playwright (浏览器) 和 PyAutoGUI (桌面) 两种模式
"""
import time
import logging
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np

from core.execution.actions import Action, ActionType

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """执行器基类"""
    
    @abstractmethod
    def execute(self, action: Action) -> bool:
        """执行动作"""
        pass
    
    @abstractmethod
    def take_screenshot(self) -> np.ndarray:
        """获取当前截图"""
        pass
    
    @abstractmethod
    def close(self):
        """关闭资源"""
        pass


class PlaywrightExecutor(BaseExecutor):
    """
    Playwright执行器
    用于浏览器自动化
    """
    
    def __init__(self, headless: bool = False, viewport_width: int = 1280, viewport_height: int = 800):
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.playwright = None
        self.browser = None
        self.page = None
        self._started = False
    
    def start(self, url: str = "about:blank"):
        """启动浏览器"""
        from playwright.sync_api import sync_playwright
        import cv2
        
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page(
            viewport={"width": self.viewport_width, "height": self.viewport_height}
        )
        self.page.goto(url)
        self._started = True
        time.sleep(2)  # 等待页面初始加载
        logger.info(f"浏览器已启动，访问: {url}")
    
    def navigate(self, url: str):
        """导航到指定URL"""
        if not self._started:
            self.start(url)
        else:
            self.page.goto(url)
            time.sleep(1)
    
    def execute(self, action: Action) -> bool:
        """
        执行动作
        
        Args:
            action: 要执行的动作
            
        Returns:
            bool: 执行是否成功
        """
        if not self._started:
            raise RuntimeError("执行器未启动，请先调用 start() 方法")
        
        try:
            if action.action_type == ActionType.CLICK_LEFT:
                x, y = action.coordinates
                self.page.mouse.click(x, y)
                logger.info(f"执行左键点击: ({x}, {y})")
                time.sleep(0.5)
                
            elif action.action_type == ActionType.CLICK_RIGHT:
                x, y = action.coordinates
                self.page.mouse.click(x, y, button="right")
                logger.info(f"执行右键点击: ({x}, {y})")
                time.sleep(0.5)
                
            elif action.action_type == ActionType.SCROLL_UP:
                self.page.mouse.wheel(0, -action.scroll_amount)
                logger.info(f"向上滚动: {action.scroll_amount}px")
                time.sleep(0.5)
                
            elif action.action_type == ActionType.SCROLL_DOWN:
                self.page.mouse.wheel(0, action.scroll_amount)
                logger.info(f"向下滚动: {action.scroll_amount}px")
                time.sleep(0.5)
                
            elif action.action_type == ActionType.TYPE:
                if action.coordinates:
                    x, y = action.coordinates
                    self.page.mouse.click(x, y)
                    time.sleep(0.2)
                self.page.keyboard.type(action.text, delay=50)
                logger.info(f"键入文本: '{action.text[:30]}..'" if len(action.text) > 30 else f"键入文本: '{action.text}'")
                time.sleep(0.5)
                
            elif action.action_type == ActionType.WAIT:
                logger.info(f"等待: {action.wait_time}秒")
                time.sleep(action.wait_time)
                
            elif action.action_type == ActionType.STOP:
                logger.info("收到停止指令")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            return False
    
    def take_screenshot(self) -> np.ndarray:
        """获取当前页面截图"""
        import cv2
        
        if not self._started:
            raise RuntimeError("执行器未启动")
        
        screenshot_bytes = self.page.screenshot()
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    def get_page_info(self) -> dict:
        """获取页面基本信息"""
        if not self._started:
            return {}
        
        return {
            "url": self.page.url,
            "title": self.page.title(),
            "viewport": {"width": self.viewport_width, "height": self.viewport_height}
        }
    
    def close(self):
        """关闭浏览器"""
        if self.browser:
            self.browser.close()
            logger.info("浏览器已关闭")
        if self.playwright:
            self.playwright.stop()
        self._started = False


class PyAutoGUIExecutor(BaseExecutor):
    """
    PyAutoGUI执行器
    用于桌面应用自动化
    """
    
    def __init__(self):
        try:
            import pyautogui
            self.pyautogui = pyautogui
            # 设置安全模式
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.1
        except ImportError:
            raise ImportError("请安装 pyautogui: pip install pyautogui")
    
    def execute(self, action: Action) -> bool:
        """执行桌面动作"""
        try:
            if action.action_type == ActionType.CLICK_LEFT:
                x, y = action.coordinates
                self.pyautogui.click(x, y)
                logger.info(f"执行左键点击: ({x}, {y})")
                
            elif action.action_type == ActionType.CLICK_RIGHT:
                x, y = action.coordinates
                self.pyautogui.rightClick(x, y)
                logger.info(f"执行右键点击: ({x}, {y})")
                
            elif action.action_type == ActionType.SCROLL_UP:
                # pyautogui的scroll正值向上滚动
                clicks = action.scroll_amount // 100
                self.pyautogui.scroll(clicks)
                logger.info(f"向上滚动: {clicks}次")
                
            elif action.action_type == ActionType.SCROLL_DOWN:
                clicks = action.scroll_amount // 100
                self.pyautogui.scroll(-clicks)
                logger.info(f"向下滚动: {clicks}次")
                
            elif action.action_type == ActionType.TYPE:
                if action.coordinates:
                    x, y = action.coordinates
                    self.pyautogui.click(x, y)
                    time.sleep(0.2)
                self.pyautogui.write(action.text, interval=0.05)
                logger.info(f"键入文本: '{action.text}'")
                
            elif action.action_type == ActionType.WAIT:
                logger.info(f"等待: {action.wait_time}秒")
                time.sleep(action.wait_time)
                
            elif action.action_type == ActionType.STOP:
                logger.info("收到停止指令")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            return False
    
    def take_screenshot(self) -> np.ndarray:
        """获取屏幕截图"""
        import cv2
        
        screenshot = self.pyautogui.screenshot()
        img = np.array(screenshot)
        # PIL返回RGB，转换为BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def close(self):
        """PyAutoGUI不需要特殊关闭"""
        pass


class ActionExecutor:
    """
    动作执行器工厂类
    根据模式创建对应的执行器
    """
    
    def __init__(self, mode: str = "browser", **kwargs):
        """
        初始化执行器
        
        Args:
            mode: "browser" 或 "desktop"
            **kwargs: 传递给具体执行器的参数
        """
        self.mode = mode
        
        if mode == "browser":
            self.executor = PlaywrightExecutor(**kwargs)
        elif mode == "desktop":
            self.executor = PyAutoGUIExecutor(**kwargs)
        else:
            raise ValueError(f"不支持的执行模式: {mode}")
    
    def start(self, url: str = None):
        """启动执行器（仅浏览器模式）"""
        if self.mode == "browser" and url:
            self.executor.start(url)
    
    def execute(self, action: Action) -> bool:
        """执行动作"""
        return self.executor.execute(action)
    
    def take_screenshot(self) -> np.ndarray:
        """获取截图"""
        return self.executor.take_screenshot()
    
    def close(self):
        """关闭执行器"""
        self.executor.close()

