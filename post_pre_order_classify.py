import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch

# 1. 加载数据（你需要将CSV文件路径改成你本地的路径）
df = pd.read_csv("data/order_classify_dataset_2000.csv")  # 包含 text, label 两列
dataset = Dataset.from_pandas(df)

# 2. 加载 tokenizer 和模型
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. 分词预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 4. 划分训练集和验证集
encoded_dataset = encoded_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["test"]

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./bert_classify_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1
)

# 6. 自定义评估指标（accuracy + f1）
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=1)
    labels = torch.tensor(p.label_ids)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# 7. 初始化 Trainer 并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
