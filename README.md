# Agent

## Introduction
初步打算基于OS-Atlas-Base-7B、OS-Atlas-Pro-7B、qwenvl、gpt4o等模型，验证各种agent的实现方式，实现基于gui的agent。
所有的模型都是基于openai api的形式调用，开源模型通过vllm部署
自行部署后可以通过
mv grounding_model_demo/config.yaml.bak grounding_model_demo/config.yaml
修改配置文件中的api_key和model_name来使用自己的模型

TODO LIST:
1、grounding web demo (done)
2、test OS-Atlas-Pro-7B的任务规划性能
3、OS-Atlas-Base-7B、OS-Atlas-Pro-7B、qwenvl三个模型组成 规划-感知-反思智能体系统，测试效果
4、虚拟机直接映射到浏览器上的demo，类似于claude computer use demo
