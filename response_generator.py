import requests
import os

from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的变量
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "Qwen/QwQ-32B"

def generate_prompt(comment, aspects, emotion, history=None):
    aspect_str = "、".join(aspects)
    tone = "亲切、真诚、温和" if emotion == "积极" else "诚恳、安抚、真诚"
    history_text = history + "\n---\n" if history else ""

    return f"""你是一名外卖平台的专业客服，请你以{tone}的语气，用一句话回复用户的评论，表达理解、感谢或安抚。
    请严格遵守以下规则：
    - 仅表达理解、感谢或安抚；**不得**涉及退款、调查、处理、转单等操作类承诺
    - **仅回复一句话，控制在 50 个字以内**
    - **禁止编号、分点或模板化措辞**
    - 语言必须自然、有温度，贴近真实客服说话风格，避免太官方或生硬

    【示例】
    用户评论：今天的菜很难吃！
    情绪：消极
    涉及方面：菜品口味
    客服回复：真的很抱歉今天的菜没有让您满意，您的反馈我们一定会认真记录并努力改进。

    ---
    {history_text}
    用户评论：{comment}
    情绪：{emotion}
    涉及方面：{aspect_str}
    请生成一句客服回复：
    """


def generate_response(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.85,
        "top_k": 20,
        "frequency_penalty": 0.8,
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        reply = content.split("。")[0] + "。"
        print("🧾 原始输出：", content)
        print("✅ 最终回复：", reply)
        return reply
    except Exception as e:
        print("❌ 调用 Qwen3 API 出错：", e)
        return "系统暂时繁忙，请稍后再试～"
