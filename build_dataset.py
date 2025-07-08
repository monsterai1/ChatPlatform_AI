import pandas as pd
import json
from collections import defaultdict

# 读取评论数据
df = pd.read_csv("data/waimai_10k.csv")
df = df.dropna(subset=["review", "label"])  # 清除空值

# 提取评论和情绪标签
reviews = df["review"].tolist()
labels = df["label"].tolist()

# 主体分类规则（平台/骑手/商家）
subject_rules = {
    "骑手": ["骑手", "外卖员", "小哥", "配送员", "快递", "送餐"],
    "商家": ["商家", "店家", "店员", "客服", "老板"],
    "平台": ["平台", "百度", "美团", "饿了么"]
}

# 分类关键词规则（带正负）
keyword_rules = {
    "配送速度": {
        "positive": ["很快", "准时", "及时", "飞快", "速度快", "送餐快"],
        "negative": ["晚", "慢", "久", "等", "延迟", "超时", "太久"]
    },
    "配送完整性": {
        "positive": ["齐全", "都送了", "一样不少"],
        "negative": ["没送", "漏送", "没有收到", "少了", "缺了"]
    },
    "味道评价": {
        "positive": ["好吃", "味道好", "香", "可口", "喜欢吃"],
        "negative": ["太辣", "很辣", "太咸", "太淡", "口味重", "太油", "难吃", "臭"]
    },
    "分量评价": {
        "positive": ["分量足", "量大", "够吃", "吃撑了"],
        "negative": ["分量少", "太少", "量太小", "不够吃", "吃不饱"]
    },
    "骑手服务": {
        "positive": ["态度好", "很有礼貌", "亲切", "服务好", "送得快", "打电话提醒"],
        "negative": ["态度差", "不耐烦", "语气差", "发火", "挂电话", "凶", "恶劣"]
    },
    "商家服务": {
        "positive": ["客服好", "老板热情", "服务周到", "店家负责", "积极回应"],
        "negative": ["客服差", "态度差", "不理人", "退单", "取消", "说话凶"]
    },
    "包装情况": {
        "positive": ["包装好", "密封好", "干净整洁", "很卫生"],
        "negative": ["包装破", "洒了", "撒了", "渗漏", "盒子烂", "汤洒"]
    },
    "餐品准确性": {
        "positive": ["送对了", "没有送错", "和点的一样"],
        "negative": ["送错", "点错", "不是我点的", "给错了"]
    },
    "配送方式": {
        "positive": ["送上楼", "送到门口", "亲自送到"],
        "negative": ["不送上楼", "让我下楼拿", "没送到门口"]
    },
    "商家接单情况": {
        "positive": ["接单快", "及时接单", "营业中"],
        "negative": ["接单失败", "打烊了", "拒单", "取消了订单"]
    },
    "配送时机": {
        "positive": ["刚好饭点", "按时送达"],
        "negative": ["太早送来", "没到饭点就送"]
    }
}

# 自动摘要生成函数（根据情感生成回答）
def generate_answer(category, emotion, examples):
    if emotion == "积极":
        intro = f"您好～关于“{category}”方面，用户整体反馈不错，"
        if category == "配送速度":
            detail = "很多顾客表示送餐及时、效率很高～"
        elif category == "味道评价":
            detail = "大家普遍觉得菜品可口，味道好～"
        elif category == "分量评价":
            detail = "许多用户提到分量足、性价比高～"
        elif category == "骑手服务":
            detail = "配送员服务态度好，令人满意哦～"
        elif category == "包装情况":
            detail = "包装整洁密封好，让人放心～"
        elif category == "商家服务":
            detail = "商家服务周到，有问必答～"
        else:
            detail = "体验整体较好，欢迎您放心下单～"
        return intro + detail
    elif emotion == "消极":
        intro = f"很抱歉，有部分用户对“{category}”方面提出过反馈，"
        if category == "配送速度":
            detail = "如配送延迟或等待时间较长，店铺正在持续优化～"
        elif category == "味道评价":
            detail = "如口味偏重或不合预期等问题，商家已收到建议～"
        elif category == "分量评价":
            detail = "如量偏少或不够吃的情况，也有用户反映过～"
        elif category == "骑手服务":
            detail = "如态度不佳、联系不畅等，平台和商家都在持续改进～"
        elif category == "包装情况":
            detail = "部分用户反馈包装不牢或漏汤，我们已提醒店家注意～"
        elif category == "餐品准确性":
            detail = "有用户提到送错餐，我们建议确认后收货～"
        else:
            detail = "建议下单前查看评论，选择服务口碑较好的时间段～"
        return intro + detail
    else:
        return "这类问题暂无统一反馈，如有具体问题欢迎留言或联系商家～"


# 生成三元组和FAQ结构
tuple_list = []
faq_map = defaultdict(list)

for review, label in zip(reviews, labels):
    emotion_label = "positive" if label == 1 else "negative"
    matched = False
    aspects = []

    for category, emo_dict in keyword_rules.items():
        has_pos = any(k in review for k in emo_dict.get("positive", []))
        has_neg = any(k in review for k in emo_dict.get("negative", []))

        if has_pos and has_neg:
            continue

        if (emotion_label == "positive" and has_pos) or (emotion_label == "negative" and has_neg):
            aspects.append(category)
            matched = True

    if not matched:
        faq_map["其他"].append(review)
        continue

    subject = "平台"
    for sub, sub_kw in subject_rules.items():
        if any(k in review for k in sub_kw):
            subject = sub
            break

    for asp in aspects:
        tuple_list.append((subject, asp, "积极" if emotion_label == "positive" else "消极"))
        faq_map[f"{asp}（{emotion_label}）"].append(review)

# 保存三元组
with open("data/triplets.json", "w", encoding="utf-8") as f:
    json.dump([{"subject": t[0], "aspect": t[1], "sentiment": t[2]} for t in tuple_list], f, ensure_ascii=False, indent=2)

# 构造FAQ结构（含 answer 字段）
faq_list = []
for cat_emotion, comments in faq_map.items():
    if cat_emotion == "其他":
        faq_list.append({
            "category": "其他",
            "emotion": "未分类",
            "question": "未分类的其他问题",
            "answer": "这类问题没有明显的情感或方面归属，建议人工进一步分析。",
            "example_comments": comments[:5]
        })
    else:
        cat, emo = cat_emotion.replace("（positive）", "|positive").replace("（negative）", "|negative").split("|")
        emo_label = "积极" if emo == "positive" else "消极"
        faq_list.append({
            "category": cat,
            "emotion": emo_label,
            "question": f"{cat}（{emo_label}）怎么办？",
            "answer": generate_answer(cat, emo_label, comments[:5]),
            "example_comments": comments[:5]
        })

# 保存 FAQ
with open("data/faq_seed.json", "w", encoding="utf-8") as f:
    json.dump(faq_list, f, ensure_ascii=False, indent=2)

print(f"✅ 构造完成，共三元组数：{len(tuple_list)}，问答对数：{len(faq_list)}")
