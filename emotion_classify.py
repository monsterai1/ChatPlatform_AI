import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

project_root = os.path.dirname(os.path.abspath(__file__))

# 分开加载路径
tokenizer_path = os.path.join(project_root, "model", "emotion_waimai_model")
model_path = os.path.join(project_root, "model", "emotion_waimai_model", "checkpoint-2698")

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    top_k=None
)

label_map = model.config.id2label  # e.g., {0: "negative (stars 1, 2 and 3)", 1: "positive (stars 4 and 5)"}

# 中文映射
label_zh_map = {
    "negative (stars 1, 2 and 3)": "消极",
    "positive (stars 4 and 5)": "积极"
}


def emotion_analyze(text):
    result = classifier(text)[0]
    top_result = max(result, key=lambda x: x['score'])
    label = top_result['label']
    label_zh = label_zh_map.get(label, label)
    score = round(top_result['score'], 4)
    print(label_zh, score)
    return label_zh, score

if __name__ == "__main__":
    emotion_analyze("今天送餐都迟到了！")
