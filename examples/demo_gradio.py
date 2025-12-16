"""
Gradio Web UI 示例
提供一个Web界面来控制Agent
"""
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import threading
import queue
from typing import Generator
import numpy as np

from core import create_agent, get_config
from core.execution.actions import Action


class AgentUI:
    """Agent Web UI 控制器"""
    
    def __init__(self):
        self.agent = None
        self.is_running = False
        self.log_queue = queue.Queue()
        self.current_screenshot = None
    
    def log(self, message: str):
        """添加日志消息"""
        self.log_queue.put(message)
    
    def step_callback(self, step: int, action: Action, perception: dict, success: bool):
        """每步执行后的回调"""
        status = "✓" if success else "✗"
        self.log(f"Step {step} {status}: {action}")
        if action.thought:
            self.log(f"  └ 推理: {action.thought}")
    
    def run_task(self, task: str, url: str, max_steps: int, headless: bool):
        """执行任务"""
        if self.is_running:
            yield "⚠️ 任务正在运行中...", None
            return
        
        self.is_running = True
        self.log(f"🚀 开始任务: {task}")
        self.log(f"   URL: {url}")
        self.log(f"   最大步数: {max_steps}")
        
        try:
            self.agent = create_agent(mode="browser", headless=headless)
            self.agent.set_step_callback(self.step_callback)
            
            trajectory = self.agent.run(
                task=task,
                start_url=url,
                max_steps=int(max_steps)
            )
            
            self.log("\n" + "=" * 40)
            self.log(f"✅ 任务完成!")
            self.log(f"   总步数: {len(trajectory)}")
            self.log(f"   状态: {trajectory.status}")
            
        except Exception as e:
            self.log(f"\n❌ 执行出错: {str(e)}")
        finally:
            self.is_running = False
        
        # 收集所有日志
        logs = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get())
        
        yield "\n".join(logs), None
    
    def stop_task(self):
        """停止任务"""
        if self.agent and self.is_running:
            self.agent.stop()
            return "🛑 已发送停止信号"
        return "没有正在运行的任务"
    
    def get_config_info(self):
        """获取配置信息"""
        config = get_config()
        return f"""
配置信息:
- Base URL: {config.base_url}
- Vision Model: {config.vision_model}
- Reasoning Model: {config.reasoning_model}
- Max Steps: {config.max_steps}
"""


def create_ui():
    """创建Gradio界面"""
    controller = AgentUI()
    
    with gr.Blocks(title="GUI ReAct Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 GUI ReAct Agent
        基于纯视觉感知的GUI自动化系统
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                task_input = gr.Textbox(
                    label="任务描述",
                    placeholder="例如：在百度上搜索'Python教程'",
                    lines=2
                )
                url_input = gr.Textbox(
                    label="起始URL",
                    value="https://www.baidu.com"
                )
                
                with gr.Row():
                    max_steps = gr.Slider(
                        label="最大步数",
                        minimum=5,
                        maximum=50,
                        value=15,
                        step=1
                    )
                    headless = gr.Checkbox(
                        label="无头模式",
                        value=False
                    )
                
                with gr.Row():
                    run_btn = gr.Button("🚀 开始执行", variant="primary")
                    stop_btn = gr.Button("🛑 停止", variant="stop")
            
            with gr.Column(scale=1):
                config_display = gr.Textbox(
                    label="配置信息",
                    value=controller.get_config_info(),
                    lines=6,
                    interactive=False
                )
        
        with gr.Row():
            output_log = gr.Textbox(
                label="执行日志",
                lines=15,
                max_lines=30,
                interactive=False
            )
        
        with gr.Row():
            screenshot_display = gr.Image(
                label="当前截图",
                type="numpy"
            )
        
        # 绑定事件
        run_btn.click(
            fn=controller.run_task,
            inputs=[task_input, url_input, max_steps, headless],
            outputs=[output_log, screenshot_display]
        )
        
        stop_btn.click(
            fn=controller.stop_task,
            outputs=[output_log]
        )
        
        # 示例任务
        gr.Examples(
            examples=[
                ["在百度上搜索'Python教程'", "https://www.baidu.com", 10],
                ["在Google上搜索'机器学习'", "https://www.google.com", 15],
                ["在Bing上搜索'OpenAI'", "https://www.bing.com", 10],
            ],
            inputs=[task_input, url_input, max_steps]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False)

