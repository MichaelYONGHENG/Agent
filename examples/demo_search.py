"""
示例：在搜索引擎执行搜索任务
"""
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from core import create_agent


def demo_google_search():
    """Google搜索示例"""
    agent = create_agent(mode="browser", headless=False)
    
    task = "在Google上搜索'Python机器学习教程'，然后查看第一个搜索结果"
    
    trajectory = agent.run(
        task=task,
        start_url="https://www.google.com",
        max_steps=15
    )
    
    return trajectory


def demo_baidu_search():
    """百度搜索示例"""
    agent = create_agent(mode="browser", headless=False)
    
    task = "在百度上搜索'Python入门教程'，并点击第一个搜索结果"
    
    trajectory = agent.run(
        task=task,
        start_url="https://www.baidu.com",
        max_steps=15
    )
    
    return trajectory


def demo_bing_search():
    """Bing搜索示例"""
    agent = create_agent(mode="browser", headless=False)
    
    task = "在Bing上搜索'OpenAI GPT-4'，查看搜索结果"
    
    trajectory = agent.run(
        task=task,
        start_url="https://www.bing.com",
        max_steps=15
    )
    
    return trajectory


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="搜索引擎演示")
    parser.add_argument(
        "--engine", "-e",
        choices=["google", "baidu", "bing"],
        default="baidu",
        help="选择搜索引擎"
    )
    
    args = parser.parse_args()
    
    if args.engine == "google":
        demo_google_search()
    elif args.engine == "baidu":
        demo_baidu_search()
    elif args.engine == "bing":
        demo_bing_search()

