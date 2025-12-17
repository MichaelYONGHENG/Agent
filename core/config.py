"""
统一配置管理模块
通过 .env 文件管理所有配置
支持多个模型提供商（阿里云、Claude等）
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional


# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class AliyunConfig:
    """阿里云 Qwen 模型配置（用于视觉感知）"""
    api_key: str = None
    base_url: str = None
    vision_model: str = None      # 视觉感知模型
    grounding_model: str = None   # 元素定位模型
    
    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("ALIYUN_API_KEY")
        self.base_url = self.base_url or os.getenv("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.vision_model = self.vision_model or os.getenv("QWEN3_VL_PLUS_MODEL_NAME", "qwen-vl-plus")
        self.grounding_model = self.grounding_model or os.getenv("QWEN3_VL_PLUS_MODEL_NAME", "qwen-vl-plus")
    
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class ClaudeConfig:
    """Claude 模型配置（用于主Agent推理）"""
    api_key: str = None
    base_url: str = None
    reasoning_model: str = None   # 推理规划模型
    
    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("CLAUDE_API_KEY")
        self.base_url = self.base_url or os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1")
        self.reasoning_model = self.reasoning_model or os.getenv("CLAUDE45_MODEL_NAME", "claude-sonnet-4-20250514")
    
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class Config:
    """
    Agent 统一配置类
    
    支持两套模型配置：
    - 感知模型（Perception）: 使用阿里云 Qwen3-VL-Plus
    - 推理模型（Planning）: 使用 Claude
    """
    
    # 阿里云配置（感知模块）
    aliyun: AliyunConfig = field(default_factory=AliyunConfig)
    
    # Claude配置（推理模块）
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    
    # Agent 运行配置
    max_steps: int = 50
    screenshot_interval: float = 1.0
    
    def __post_init__(self):
        """从环境变量加载配置"""
        # 初始化子配置
        if not isinstance(self.aliyun, AliyunConfig):
            self.aliyun = AliyunConfig()
        if not isinstance(self.claude, ClaudeConfig):
            self.claude = ClaudeConfig()
        
        # Agent配置
        self.max_steps = int(os.getenv("MAX_STEPS", self.max_steps))
        self.screenshot_interval = float(os.getenv("SCREENSHOT_INTERVAL", self.screenshot_interval))
        
        # 验证配置
        self._validate()
    
    def _validate(self):
        """验证配置完整性"""
        errors = []
        
        if not self.aliyun.is_configured():
            errors.append("阿里云API密钥未配置。请在.env文件中设置 ALIYUN_API_KEY")
        
        if not self.claude.is_configured():
            errors.append("Claude API密钥未配置。请在.env文件中设置 CLAUDE_API_KEY")
        
        if errors:
            raise ValueError("\n".join(errors))
    
    # 便捷属性访问（保持向后兼容）
    @property
    def perception_api_key(self) -> str:
        """感知模块API密钥"""
        return self.aliyun.api_key
    
    @property
    def perception_base_url(self) -> str:
        """感知模块API地址"""
        return self.aliyun.base_url
    
    @property
    def vision_model(self) -> str:
        """视觉感知模型名称"""
        return self.aliyun.vision_model
    
    @property
    def grounding_model(self) -> str:
        """元素定位模型名称"""
        return self.aliyun.grounding_model
    
    @property
    def reasoning_api_key(self) -> str:
        """推理模块API密钥"""
        return self.claude.api_key
    
    @property
    def reasoning_base_url(self) -> str:
        """推理模块API地址"""
        return self.claude.base_url
    
    @property
    def reasoning_model(self) -> str:
        """推理规划模型名称"""
        return self.claude.reasoning_model
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  # 感知模块（阿里云 Qwen）\n"
            f"  perception_base_url={self.perception_base_url},\n"
            f"  vision_model={self.vision_model},\n"
            f"  grounding_model={self.grounding_model},\n"
            f"\n"
            f"  # 推理模块（Claude）\n"
            f"  reasoning_base_url={self.reasoning_base_url},\n"
            f"  reasoning_model={self.reasoning_model},\n"
            f"\n"
            f"  # Agent配置\n"
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


def reset_config():
    """重置全局配置（用于测试）"""
    global _config_instance
    _config_instance = None
