"""
统一配置管理模块
通过 .env 文件管理所有配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass


# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class Config:
    """Agent 配置类"""
    
    # API 配置
    api_key: str = None
    base_url: str = None
    
    # 模型配置
    vision_model: str = None
    grounding_model: str = None
    reasoning_model: str = None
    
    # Agent 配置
    max_steps: int = 50
    screenshot_interval: float = 1.0
    
    def __post_init__(self):
        """从环境变量加载配置"""
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = self.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        self.vision_model = self.vision_model or os.getenv("VISION_MODEL", "gpt-4o")
        self.grounding_model = self.grounding_model or os.getenv("GROUNDING_MODEL", "gpt-4o")
        self.reasoning_model = self.reasoning_model or os.getenv("REASONING_MODEL", "gpt-4o")
        
        self.max_steps = int(os.getenv("MAX_STEPS", self.max_steps))
        self.screenshot_interval = float(os.getenv("SCREENSHOT_INTERVAL", self.screenshot_interval))
        
        if not self.api_key:
            raise ValueError("API密钥未配置。请在.env文件中设置OPENAI_API_KEY")
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  base_url={self.base_url},\n"
            f"  vision_model={self.vision_model},\n"
            f"  grounding_model={self.grounding_model},\n"
            f"  reasoning_model={self.reasoning_model},\n"
            f"  max_steps={self.max_steps}\n"
            f")"
        )


# 全局配置实例
_config_instance = None


def get_config() -> Config:
    """获取全局配置实例（单例模式）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

