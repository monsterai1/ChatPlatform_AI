import requests
from langchain.schema import HumanMessage, AIMessage
from emotion_classify import emotion_analyze
from faq_vector_preorder import query_preorder_faq
from aspect_detector import get_aspects
from response_generator import generate_prompt, generate_response
from transformers import pipeline

# 多轮对话历史
history_messages = []

# 餐前/餐后分类器
order_stage_classifier = pipeline(
    "text-classification",
    model="./bert_classify_output/checkpoint-75",
    tokenizer="./bert_classify_output/checkpoint-75"
)

# 构造历史上下文 Prompt
def build_history_text(messages, max_rounds=2):
    history = messages[-max_rounds * 2:] if len(messages) >= max_rounds * 2 else messages
    lines = []
    i = 0
    while i < len(history) - 1:
        if isinstance(history[i], HumanMessage) and isinstance(history[i + 1], AIMessage):
            user_msg = history[i].content
            ai_msg = history[i + 1].content
            lines.append(f"用户：{user_msg}\n客服：{ai_msg}")
            i += 2
        else:
            i += 1
    return "\n".join(lines)

# 判断是否为餐前评论
def is_preorder_query(text):
    result = order_stage_classifier(text)[0]
    print(f"[阶段判断] label: {result['label']} | score: {round(result['score'], 4)} | 文本: {text}")
    return result["label"] == "LABEL_0"

# 判断是否需要转人工
def need_manual_intervention(text, emotion):
    bad_keywords = ["差评", "投诉", "再也不", "恶心", "一次最差", "想退款"]
    return emotion == "消极" and any(k in text for k in bad_keywords)

# 主入口函数
def generate_reply(user_input):
    global history_messages

    # 判断是否餐前问询
    if is_preorder_query(user_input):
        answer = query_preorder_faq(user_input)
        history_messages.append(HumanMessage(content=user_input))
        history_messages.append(AIMessage(content=answer))
        return answer

    # 情绪识别
    emotion, score = emotion_analyze(user_input)
    print(f"[情绪分析] → {emotion} | 置信度：{round(score, 4)}")

    # 情绪强烈或敏感关键词 → 转人工
    if need_manual_intervention(user_input, emotion) or (emotion == "消极" and score < 0.6):
        return "很抱歉给您带来不便，系统已记录问题，已为您转接人工客服处理～"

    # 方面识别
    aspects = get_aspects(user_input)

    # 构造多轮上下文
    history_text = build_history_text(history_messages)

    # 构造 Prompt & 生成回复
    prompt = generate_prompt(user_input, aspects, emotion, history=history_text)
    reply = generate_response(prompt)

    # 更新对话历史
    history_messages.append(HumanMessage(content=user_input))
    history_messages.append(AIMessage(content=reply))

    return reply

# 示例运行
if __name__ == "__main__":
    while True:
        comment = input("请输入评论内容（输入 q 退出）：").strip()
        if comment.lower() == "q":
            break
        print(generate_reply(comment))
