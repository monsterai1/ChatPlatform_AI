# train_emotion.py
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
import torch

# 加载数据
df = pd.read_csv("data/waimai_10k.csv")  
df = df.rename(columns={"review": "text", "label": "labels"}) 
dataset = Dataset.from_pandas(df)

# 加载模型和 tokenizer
model_name = "uer/roberta-base-finetuned-jd-binary-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 分词预处理
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",    
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

# 划分训练 / 验证集
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 设置训练参数（适配 CPU）
training_args = TrainingArguments(
    output_dir="model/emotion_waimai_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="model/emotion_waimai_model/logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("model/emotion_waimai_model")
tokenizer.save_pretrained("model/emotion_waimai_model")
