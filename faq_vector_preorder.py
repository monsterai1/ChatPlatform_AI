# faq_preorder_vector.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


try:
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
    db = Chroma(persist_directory="./chroma/chroma_faq_preorder_db", embedding_function=embedding_model)
    print("✅ 成功加载 Chroma 向量库！")
except Exception as e:
    print("❌ 加载 Chroma 向量库失败：", e)
    db = None


def query_preorder_faq(query: str):
    print(f"🟡 [Debug] query_preorder_faq 被调用，用户输入：{query}")
    result = db.similarity_search_with_score(query, k=1)
    print(f"[Debug] 匹配 raw 结果: {result}")

    if result:
        doc, score = result[0]
        print(f"[Debug] 匹配分数: {score:.4f}, 匹配问题: {doc.page_content}")
        return doc.metadata.get("answer", "您好，这是相关信息～")

    print(f"[Debug] 无FAQ命中，返回默认回复")
    return "很抱歉，未找到相关信息，请联系人工客服处理。"

if __name__ == "__main__":
    print("📌 正在测试 FAQ 查询接口...")

    # 手动构造一个测试问题
    test_query = "可以微信支付吗"

    # 调用函数
    answer = query_preorder_faq(test_query)

    print(f"✅ 最终返回结果: {answer}")



