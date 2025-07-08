import requests
from langchain.schema import HumanMessage, AIMessage
from emotion_classify import emotion_analyze
from faq_vector_preorder import query_preorder_faq

# 硅基流动 Qwen3 模型 API 配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = ""  
MODEL_NAME = "Qwen/QwQ-32B"

# 多轮对话历史
history_messages = []

# 餐前/餐后分类器
from transformers import pipeline
order_stage_classifier = pipeline("text-classification", model="./bert_classify_output/checkpoint-75", tokenizer="./bert_classify_output/checkpoint-75")

# 构造历史上下文 Prompt
def build_history_text(messages, max_rounds=2):
    history = messages[-max_rounds * 2:] if len(messages) >= max_rounds * 2 else messages
    lines = []
    i = 0
    while i < len(history) - 1:
        if isinstance(history[i], HumanMessage) and isinstance(history[i+1], AIMessage):
            user_msg = history[i].content
            ai_msg = history[i+1].content
            lines.append(f"用户：{user_msg}\n客服：{ai_msg}")
            i += 2
        else:
            i += 1
    return "\n".join(lines)

# 评论方面抽取
def get_aspects(text):
    aspect_map = {
        "配送速度": ["送货", "迟到", "慢", "快", "准时", "及时", "飞快", "速度", "迅速", "提前", "超时"],
        "骑手态度": ["态度", "人很好", "有礼貌", "提醒", "语气", "挂电话", "服务", "爬楼梯"],
        "商家态度": ["老板", "店家"],
        "包装情况": ["包装", "漏", "洒", "密封", "整洁", "卫生", "盒子破"],
        "菜品味道": ["味道", "难吃", "好吃", "口味", "不新鲜", "香", "辣", "淡", "咸", "油", "腻"],
        "分量性价比": ["量", "实在", "性价比", "吃不饱"],
        "订单问题": ["发票", "备注", "餐具", "漏送"],
        "餐品准确性": ["送错", "点错", "不是我点的"],
        "配送方式": ["送上楼", "下楼拿"],
        "商家接单情况": ["拒单", "取消", "接单"],
        "平台": ["优惠"]
    }
    matched = []
    for aspect, keywords in aspect_map.items():
        if any(k in text for k in keywords):
            matched.append(aspect)
    return matched or ["其他"]

# 判断是否餐前
def is_preorder_query(text):
    result = order_stage_classifier(text)[0]
    print(f"[阶段判断] label: {result['label']} | score: {round(result['score'], 4)} | 文本: {text}")
    return result["label"] == "LABEL_0"

# 转人工关键词
def need_manual_intervention(text, emotion):
    bad_keywords = ["差评", "投诉", "再也不", "恶心", "一次最差", "想退款"]
    return emotion == "消极" and any(k in text for k in bad_keywords)

# 构造 Prompt
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

# 主入口函数
def generate_reply(user_input):
    global history_messages

    if is_preorder_query(user_input):
        answer = query_preorder_faq(user_input)
        history_messages.append(HumanMessage(content=user_input))
        history_messages.append(AIMessage(content=answer))
        return answer

    emotion, score = emotion_analyze(user_input)
    print(f"[情绪分析] → {emotion} | 置信度：{round(score, 4)}")

    if need_manual_intervention(user_input, emotion) or (emotion == "消极" and score < 0.6):
        return "很抱歉给您带来不便，系统已记录问题，已为您转接人工客服处理～"

    aspects = get_aspects(user_input)
    history_text = build_history_text(history_messages)
    prompt = generate_prompt(user_input, aspects, emotion, history=history_text)

    # 调用 Qwen3 API
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
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
        content = result['choices'][0]['message']['content'].strip()
        reply = content.split("。")[0] + "。"

        history_messages.append(HumanMessage(content=user_input))
        history_messages.append(AIMessage(content=reply))

        print("🧾 原始输出：", content)
        print("✅ 最终回复：", reply)
        return reply
    except Exception as e:
        print("❌ 调用 Qwen3 API 出错：", e)
        return "系统暂时繁忙，请稍后再试～"

# 示例运行
if __name__ == "__main__":
    while True:
        comment = input("请输入评论内容（输入 q 退出）：").strip()
        if comment.lower() == "q":
            break
        print(generate_reply(comment))
