import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="web_demo\grounding_model_demo\config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()
    
    def load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
    
    @property        
    def api_key(self):
        return self.config.get("api_key")
    
    @property
    def base_url(self):
        return self.config.get("base_url")
    
    @property
    def grounding_model_name(self):
        return self.config.get("grounding_model_name")
