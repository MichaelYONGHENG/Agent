# Agent

## 🌟 Introduction

This project explores the implementation of various agents leveraging models such as **OS-Atlas-Base-7B**, **OS-Atlas-Pro-7B**, **qwenvl**, and **gpt4o**. The aim is to validate agent implementation methods and develop a GUI-based agent system. Models are accessed via the OpenAI API or deployed locally using **vllm**.

### Deployment Configuration

- For open-source models:
  - Deploy using **vllm**.
  - After deployment, update the configuration file:
    ```bash
    mv grounding_model_demo/config.yaml.bak grounding_model_demo/config.yaml
    ```
  - Modify `api_key` and `model_name` in the configuration file to integrate your custom models.

---

## ✅ TODO List

### 1️⃣ Grounding Web Demo *(Done)*

- Successfully implemented a basic grounding web demo.

### 2️⃣ Task Planning Performance Evaluation

- Test **OS-Atlas-Pro-7B** for task planning efficiency and robustness.

### 3️⃣ Intelligent System Testing

- Build a **Plan-Perceive-Reflect Agent System** using the following models:
  - **Planner**: OS-Atlas-Base-7B
  - **Perception**: OS-Atlas-Pro-7B
  - **Reflection**: qwenvl
- Conduct performance and interaction tests.

### 4️⃣ Virtual Machine Browser Demo

- Develop a demo enabling a virtual machine to be directly accessed via the browser interface.
  - Similar to **Claude's computer-use demo**.


## 🛠 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MichaelYONGHENG/Agent.git
   cd Agent
   ```
2. Set up the environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your API key and model:
   ```bash
   mv grounding_model_demo/config.yaml.bak grounding_model_demo/config.yaml
   nano grounding_model_demo/config.yaml
   ```
4. Launch the demo:
   ```bash
   python grounding_model_demo/grounding_model_test.py
   ```



## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

By following this format, the README will appear professional, be user-friendly, and clearly communicate the project's goals and features.