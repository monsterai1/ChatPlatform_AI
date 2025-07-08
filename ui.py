import gradio as gr
from chat_router import generate_reply

chat_history = []

def render_chat(history):
    html = '<div style="max-width: 800px; margin: auto;">'  # 居中
    for item in history:
        if item["role"] == "user":
            html += f'''
            <div style="text-align: right; margin: 12px 0;">
                <div style="
                    display: inline-block;
                    background: #DCF8C6;
                    padding: 10px 16px;
                    border-radius: 16px;
                    max-width: 80%;
                    font-size: 16px;
                    line-height: 1.5;
                ">
                    {item["text"]}
                </div>
            </div>
            '''
        else:  # ai
            html += f'''
            <div style="text-align: left; margin: 12px 0;">
                <div style="
                    display: inline-block;
                    background: #F1F0F0;
                    padding: 10px 16px;
                    border-radius: 16px;
                    max-width: 80%;
                    font-size: 16px;
                    line-height: 1.5;
                ">
                    {item["text"]}
                </div>
            </div>
            '''
    html += '</div>'
    return html


def chat_interface(user_input):
    if not user_input.strip():
        return "", render_chat(chat_history)

    reply = generate_reply(user_input)
    chat_history.append({"role": "user", "text": user_input})
    chat_history.append({"role": "ai", "text": reply})
    return "", render_chat(chat_history)

with gr.Blocks(title="外卖客服系统") as demo:
    gr.Markdown("## 🛵 外卖平台智能客服助手")
    gr.Markdown("请输入用户评论或问题，系统将智能识别并生成客服回复。")

    with gr.Row():
        with gr.Column(scale=3):
            user_input = gr.Textbox(
                placeholder="请输入评论内容，如：今天送餐太慢了！",
                label="✍ 用户输入",
                lines=2
            )
            submit_btn = gr.Button("发送")

        with gr.Column(scale=5):
            chat_output = gr.HTML()

    submit_btn.click(fn=chat_interface, inputs=user_input, outputs=[user_input, chat_output])

if __name__ == "__main__":
    demo.launch()
