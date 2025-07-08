import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


# ✅ 加载 FAQ
with open("data/faq_preorder.json", "r", encoding="utf-8") as f:
    faq_list = json.load(f)

docs = [Document(page_content=item["question"], metadata={"answer": item["answer"]}) for item in faq_list]

# ✅ 使用 embedding 模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

# ✅ 正确构建 Chroma 向量库
db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma/chroma_faq_preorder_db"  # ✅ 持久化路径
)

# ✅ 保存成功标志
print(f"[✅] FAQ 向量库构建完成，当前共 {len(docs)} 条")
