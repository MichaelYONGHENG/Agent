# GUI ReAct Agent

基于纯视觉感知的GUI自动化Agent系统

## 🌟 简介

这是一个基于 **ReAct（Reasoning + Acting）** 范式的GUI自动化Agent系统。系统通过纯视觉感知理解界面，无需依赖DOM结构，实现跨平台的GUI自动化操作。

### 核心特点

- **纯视觉感知**：通过截图+多模态LLM理解界面，无需解析DOM
- **ReAct架构**：Thought → Action → Observation 循环
- **双模型架构**：
  - 📷 **感知模块**：阿里云 Qwen3-VL-Plus（视觉理解和元素定位）
  - 🧠 **推理模块**：Claude（任务规划和决策）
- **灵活的动作空间**：点击、滑动、键入、等待、停止
- **完整的轨迹记录**：记录每步操作，支持回溯和调试

---

## 📁 项目结构

```
Agent/
├── core/                           # 核心模块
│   ├── config.py                   # 配置管理（多模型支持）
│   ├── agent.py                    # 主Agent类（感知-规划-执行循环）
│   ├── llm/
│   │   └── client.py               # LLM客户端（ChatClient + 重试机制）
│   ├── perception/
│   │   └── vision_model.py         # 视觉感知模块（阿里云 Qwen）
│   ├── planning/
│   │   └── planner.py              # ReAct规划器（Claude）
│   ├── execution/
│   │   ├── actions.py              # 动作空间定义
│   │   └── action_executor.py      # 动作执行器（Playwright/PyAutoGUI）
│   └── memory/
│       └── trajectory.py           # 轨迹记忆
├── examples/
│   ├── demo_search.py              # 搜索引擎演示
│   └── demo_gradio.py              # Gradio Web UI
├── grounding_model_demo/           # Grounding模型测试（历史）
├── main.py                         # 入口文件
├── pyproject.toml                  # 项目配置和依赖
└── .env                            # 环境变量配置（需手动创建）
```

---

## 🛠 安装

### 1. 克隆仓库

```bash
git clone https://github.com/MichaelYONGHENG/Agent.git
cd Agent
```

### 2. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 安装 Playwright 浏览器
uv run playwright install chromium
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
# =========================================
# GUI ReAct Agent 环境配置
# =========================================

# -----------------------------------------
# 阿里云配置（感知模块：视觉理解和元素定位）
# -----------------------------------------
ALIYUN_API_KEY=your-aliyun-api-key
ALIYUN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN3_VL_PLUS_MODEL_NAME=qwen3-vl-plus

# -----------------------------------------
# Claude配置（推理模块：任务规划和决策）
# -----------------------------------------
CLAUDE_API_KEY=your-claude-api-key
CLAUDE_BASE_URL=https://api.anthropic.com/v1
CLAUDE45_MODEL_NAME=claude-sonnet-4-20250514

# -----------------------------------------
# Agent运行配置
# -----------------------------------------
MAX_STEPS=50
SCREENSHOT_INTERVAL=1.0
```

---

## 🚀 使用方法

### 命令行运行

```bash
# 查看帮助
uv run python main.py --help

# 运行演示任务
uv run python main.py --demo

# 执行自定义任务
uv run python main.py --task "在百度搜索Python教程" --url "https://www.baidu.com"

# 指定最大步数
uv run python main.py --task "在Google搜索机器学习" --url "https://www.google.com" --max-steps 15

# 无头模式（不显示浏览器窗口）
uv run python main.py --task "搜索任务" --url "https://www.bing.com" --headless
```

### Python代码调用

```python
from core import create_agent

# 创建Agent
agent = create_agent(mode="browser", headless=False)

# 执行任务
trajectory = agent.run(
    task="在百度上搜索'Python教程'并点击第一个结果",
    start_url="https://www.baidu.com",
    max_steps=20
)

# 查看结果
trajectory.print_summary()
```

### Gradio Web界面

```bash
uv run python examples/demo_gradio.py
```

---

## 🎯 动作空间

| 动作 | 说明 | 参数 |
|------|------|------|
| `click_left` | 左键点击 | `x, y` |
| `click_right` | 右键点击 | `x, y` |
| `scroll_up` | 向上滑动 | `amount`（默认300） |
| `scroll_down` | 向下滑动 | `amount`（默认300） |
| `type` | 键入文本 | `text, x, y`（坐标可选） |
| `wait` | 等待 | `seconds` |
| `stop` | 停止任务 | - |

---

## 🔧 架构说明

```
┌─────────────────────────────────────────────────────────────┐
│                        主Agent                               │
│                  （感知-规划-执行循环）                         │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   感知模块       │  │   规划模块       │  │   执行模块       │
│ VisionPerception│  │  ReActPlanner   │  │ ActionExecutor  │
│                 │  │                 │  │                 │
│ 阿里云 Qwen-VL  │  │    Claude       │  │   Playwright    │
│ - 场景理解       │  │ - 任务推理      │  │ - 点击/滑动     │
│ - 元素定位       │  │ - 动作规划      │  │ - 键入/等待     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   轨迹记忆       │
                    │   Trajectory    │
                    │ - 历史记录       │
                    │ - 截图保存       │
                    └─────────────────┘
```

### 执行流程

1. **感知（Perception）**：截取当前界面，使用 Qwen-VL 分析场景和可交互元素
2. **规划（Planning）**：Claude 基于感知结果和历史轨迹，决定下一步动作
3. **执行（Execution）**：Playwright 执行具体的GUI操作
4. **记录（Memory）**：保存每步的动作、截图和推理过程
5. **循环**：重复上述步骤直到任务完成或达到最大步数

---

## 📊 示例输出

```
============================================================
🤖 GUI ReAct Agent
============================================================
配置信息:
  📷 感知模块（阿里云 Qwen）:
     - Vision Model: qwen3-vl-plus
     - Grounding Model: qwen3-vl-plus
  🧠 推理模块（Claude）:
     - Reasoning Model: claude-sonnet-4-20250514
  ⚙️ Agent配置:
     - Max Steps: 50
============================================================

🚀 开始任务: 在百度上搜索'Python教程'

==================================================
📍 Step 1/50
==================================================
👁️  感知中...
   场景: 百度首页，包含搜索框和搜索按钮
   状态: 任务刚开始，需要输入搜索词

🧠 规划中...
   决策: Action(type, text='Python教程', coords=(640, 200))
   推理: 需要先在搜索框中输入关键词

⚡ 执行中...
   ✓ 动作执行成功

...

============================================================
📊 任务轨迹摘要
============================================================
任务: 在百度上搜索'Python教程'
状态: completed
总步数: 5
成功: 5, 失败: 0
耗时: 23.45秒
============================================================
```

---

## ✅ TODO List

- [x] 基础Grounding Web Demo
- [x] ReAct Agent架构实现
- [x] 双模型配置（感知+推理）
- [x] Playwright浏览器自动化
- [x] 轨迹记录和回溯
- [ ] PyAutoGUI桌面自动化支持
- [ ] 更多技能封装（登录、表单填写等）
- [ ] 虚拟机浏览器访问Demo
- [ ] 性能优化和错误恢复

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
