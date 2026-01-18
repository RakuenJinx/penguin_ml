from openai import OpenAI
from transformers import pipeline

import streamlit as st

st.title("Hugging Face Demo")


@st.cache_resource()
def get_model():
    return pipeline("sentiment-analysis")


nlp = get_model()
text = st.text_input("输入文本")
if text:
    result = nlp(text)
    st.write("情绪 : ", result[0]["label"])
    st.write("置信度:", result[0]["score"])
st.title("千问分析 Demo")
analyze_button = st.button("分析")
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-5fa5e2b5252244e6a66e249451c95569",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
system_message_default = "你是个有经验的sentiment分析专家，你总能给出准确的分析结果,并给出正面、负面、中性的情绪判断、0-1的置信度及简要原因"
system_message = st.text_area("请输入系统提示词", system_message_default)
if analyze_button:
    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {"role": "user", "content": f"请分析如下文本：{text}"},
    ]
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
    )
    setiment = response.choices[0].message.content.strip()
    st.write(setiment)
