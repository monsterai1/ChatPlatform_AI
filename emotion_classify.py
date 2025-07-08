# emotion_model.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # 屏蔽 transformers 加载 TensorFlow

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

project_root = os.path.dirname(os.path.abspath(__file__))  
print("project_root", project_root)
model_path = os.path.join(project_root, "model", "emotion_classifier")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    top_k=None,  
)

# 英文标签映射到英文简写
label_map = {
    "negative (stars 1, 2 and 3)": "negative",
    "positive (stars 4 and 5)": "positive"
}

# 中文情绪标签映射
label_zh_map = {
    "positive": "积极",
    "negative": "消极"
}

def emotion_analyze(text):
    result = classifier(text)[0]
    top_result = max(result, key=lambda x: x['score'])
    label_en = label_map.get(top_result['label'], top_result['label'])  # 英文标签
    label_zh = label_zh_map.get(label_en, label_en)  # 中文标签
    score = top_result['score']
    print(label_zh, round(score, 4))
    return label_zh, round(score, 4)

if __name__ == "__main__":
    emotion_analyze("饭太难吃了！")
