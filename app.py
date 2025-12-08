import streamlit as st
from openai import OpenAI
import os

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="Zeeshan ka Chatbot 🤖", page_icon="🤖", layout="centered")

# --------------------------
# Custom CSS for Amazing UI
# --------------------------
st.markdown("""
<style>

.chat-bubble-user {
    background: #16a34a;
    padding: 12px 18px;
    border-radius: 12px;
    margin: 8px;
    color: white;
    width: fit-content;
    max-width: 80%;
}

.chat-bubble-bot {
    background: #0ea5e9;
    padding: 12px 18px;
    border-radius: 12px;
    margin: 8px;
    color: white;
    width: fit-content;
    max-width: 80%;
}

.chat-container {
    padding: 10px;
    border-radius: 10px;
}

input[type=text] {
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# API Setup
# --------------------------
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# --------------------------
# Session State Setup
# --------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "input_q" not in st.session_state:
    st.session_state.input_q = ""

# --------------------------
# Simple RAG Function
# --------------------------
def rag_search(query):
    # You can later replace this with real PDF text
    return f"Based on your question, here is useful info related to '{query}'."

# --------------------------
# Generate Bot Reply
# --------------------------
def get_answer(question):
    retrieved = rag_search(question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Zeeshan ka Chatbot, helpful and friendly."},
            {"role": "user", "content": f"User question: {question}\n\nRetrieved info:\n{retrieved}"}
        ]
    )

    return response.choices[0].message["content"]

# --------------------------
# Chat Display (with cleaning)
# --------------------------
clean_chat = []
for item in st.session_state.chat:
    if isinstance(item, dict) and "sender" in item and "msg" in item:
        clean_chat.append(item)

st.session_state.chat = clean_chat

# Display messages
for item in st.session_state.chat:
    if item["sender"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>{item['msg']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>{item['msg']}</div>", unsafe_allow_html=True)

# --------------------------
# User Input
# --------------------------
st.write("### Ask Zeeshan ka Chatbot Something 👇")

user_input = st.text_input("Type your question:", value=st.session_state.input_q)

send = st.button("Send")

# --------------------------
# When user sends a message
# --------------------------
if send and user_input.strip() != "":
    st.session_state.chat.append({"sender": "user", "msg": user_input})

    bot_reply = get_answer(user_input)
    st.session_state.chat.append({"sender": "bot", "msg": bot_reply})

    st.session_state.input_q = ""   # Clear input box
    st.experimental_rerun()
