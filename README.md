# 外卖客服智能回复系统
基于真实外卖评论数据构建的智能客服系统，支持 评论情绪识别、方面抽取 与 自动生成客服回复，适用于外卖商家提升用户体验与运营效率。

## 项目系统流程
<img src="https://github.com/user-attachments/assets/77d1f9b9-6a0d-4a5a-b4ec-7abf16aa840b" width="600">

## 示例效果
用户评论：
今天的菜很难吃！
系统输出：
真的很抱歉今天的菜没有让您满意，您的反馈我们一定会认真记录并努力改进。

## 环境依赖
```bash
pip install transformers langchain chromadb gradio
```
- 推荐使用 Python 3.9+
- 推荐使用 GPU 环境运行 Qwen 模型（也可替换为任意本地中文生成模型）


## 快速启动
```bash
python chat_router.py
```
## FAQ 数据格式示例
```json
[
  {
    "question": "营业时间是多久？",
    "answer": "您好, 我们店的营业时间为11:00 - 22:00～"
  },
]
```
## License
本项目仅供学习研究使用，禁止商用。如需合作请联系作者。

