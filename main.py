"""
GUI ReAct Agent 入口文件
"""
import argparse
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from core.agent import GUIReactAgent, create_agent
from core.config import get_config


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def run_demo():
    """运行演示任务"""
    # 创建Agent
    agent = create_agent(mode="browser", headless=False)
    
    # 示例任务
    task = "在Google上搜索'Python教程'并查看搜索结果"
    
    # 执行任务
    trajectory = agent.run(
        task=task,
        start_url="https://www.google.com",
        max_steps=10
    )
    
    return trajectory


def run_task(task: str, url: str = None, max_steps: int = 20, 
             mode: str = "browser", headless: bool = False):
    """
    运行指定任务
    
    Args:
        task: 任务描述
        url: 起始URL
        max_steps: 最大步数
        mode: 执行模式 ("browser" 或 "desktop")
        headless: 是否无头模式
    """
    agent = create_agent(mode=mode, headless=headless)
    
    trajectory = agent.run(
        task=task,
        start_url=url,
        max_steps=max_steps
    )
    
    return trajectory


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GUI ReAct Agent - 基于视觉感知的GUI自动化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行演示
  python main.py --demo
  
  # 执行自定义任务
  python main.py --task "在百度搜索Python" --url "https://www.baidu.com"
  
  # 桌面模式（需要安装pyautogui）
  python main.py --task "打开记事本" --mode desktop
        """
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="运行演示任务"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="要执行的任务描述"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        default="https://www.google.com",
        help="起始URL（仅browser模式）"
    )
    parser.add_argument(
        "--max-steps", "-n",
        type=int,
        default=20,
        help="最大执行步数"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["browser", "desktop"],
        default="browser",
        help="执行模式"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="浏览器无头模式"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 显示配置信息
    config = get_config()
    print("\n" + "=" * 60)
    print("🤖 GUI ReAct Agent")
    print("=" * 60)
    print(f"配置信息:")
    print(f"  📷 感知模块（阿里云 Qwen）:")
    print(f"     - Base URL: {config.perception_base_url}")
    print(f"     - Vision Model: {config.vision_model}")
    print(f"     - Grounding Model: {config.grounding_model}")
    print(f"  🧠 推理模块（Claude）:")
    print(f"     - Base URL: {config.reasoning_base_url}")
    print(f"     - Reasoning Model: {config.reasoning_model}")
    print(f"  ⚙️ Agent配置:")
    print(f"     - Max Steps: {config.max_steps}")
    print("=" * 60 + "\n")
    
    # 执行任务
    if args.demo:
        run_demo()
    elif args.task:
        run_task(
            task=args.task,
            url=args.url,
            max_steps=args.max_steps,
            mode=args.mode,
            headless=args.headless
        )
    else:
        parser.print_help()
        print("\n💡 提示: 使用 --demo 运行演示，或使用 --task 指定任务")


if __name__ == "__main__":
    main()
