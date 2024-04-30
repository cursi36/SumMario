import sys
sys.path.append("./Agents/")
import os
os.environ['OPENAI_API_KEY'] = ""

import gradio as gr
import time
from Agents.SuperAgent import create_SuperAgent
import os


class myApp():

    def __init__(self):
        super().__init__()
        self.superAgent = create_SuperAgent()
        self.human_messages = []
        self.uploaded_files = []

    def add_text(self,history, text):
        self.human_messages.append(text)
        history = history + [(text, None)]
        return history, gr.Textbox(value="", interactive=False)

    def add_file(self,history, file):
        self.uploaded_files.append(file.name)
        history = history + [((file.name,), None)]
        return history

    def bot(self,history):

        # response = self.chatter.query(None)
        message = self.human_messages[-1]
        response = self.superAgent.run(message, to_text=True)

        # response = "**That's cool!**"
        history[-1][1] = ""
        for character in response:
            history[-1][1] += character
            time.sleep(0.01)
            yield history


if __name__ == "__main__":
    app = myApp()

    with gr.Blocks() as demo:
        gr.Markdown("Start typing below and then click **Run** to see the output.")
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            # avatar_images=(None, (os.path.join(os.path.dirname(__file__), "../Bengal_123.jpg"))),
        )

        with gr.Row():
            txt = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
                container=False,
            )
            btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

        with gr.Row():
            btn_submit = gr.Button("Submit text")

        txt_msg = txt.submit(app.add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            app.bot, chatbot, chatbot, api_name="bot_response"
        )
        txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
        file_msg = btn.upload(app.add_file, [chatbot, btn], [chatbot], queue=False).then(
            app.bot, chatbot, chatbot
        )

        res = btn_submit.click(fn=app.add_text,
                               inputs=[chatbot, txt], outputs= [chatbot, txt]).then(
            app.bot, chatbot, chatbot
        )
        res.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)


    demo.queue()
    demo.launch()