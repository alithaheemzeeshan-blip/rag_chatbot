import streamlit as st
from openai import OpenAI
import pdfplumber
import os
import time

# -----------------------
#  SETUP
# -----------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", page_icon="🤖", layout="centered")

client = OpenAI(api_key="YOUR_API_KEY")

# -----------------------
#  BEAUTIFUL CUSTOM CSS
# -----------------------
st.markdown("""
<style>

body {
    background-color: #0D0D0D;
}

.chat-bubble-user {
    background-color: #262626;
    color: white;
    padding: 12px 18px;
    border-radius: 10px;
    margin: 8px 0;
    text-align: right;
    font-size: 17px;
}

.chat-bubble-bot {
    background-color: #FFCC00;
    color: black;
    padding: 12px 18px;
    border-radius: 10px;
    margin: 8px 0;
    font-size: 17px;
}

.typing {
    color: #FFCC00;
    font-size: 16px;
    animation: blink 1s infinite;
}

@keyframes blink {
    0% {opacity: 0;}
    50% {opacity: 1;}
    100% {opacity: 0;}
}

input, textarea {
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Load PDF text
# -----------------------
def load_pdf_text():
    text = ""
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            with pdfplumber.open(f"data/{file}") as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
    return text

pdf_text = load_pdf_text()

# -----------------------
# Generate Answer
# -----------------------
def get_answer(question):
    prompt = f"""
You are Zeeshan ka Chatbot — a polite, friendly assistant.
Use the PDF information below to answer.

PDF CONTENT:
{pdf_text}

User Question: {question}

Answer clearly and in simple language.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

# -----------------------
# SESSION STATE FOR CHAT
# -----------------------
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"sender": "", "msg": ""}

# -----------------------
# CHAT UI HEADER
# -----------------------
st.markdown("<h1 style='color:white; text-align:center;'>🤖 Zeeshan ka Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#FFCC00; text-align:center;'>Smart • Simple • RAG-Based</p>", unsafe_allow_html=True)
st.write("---")

# -----------------------
# SHOW CHAT MESSAGES
# -----------------------
for item in st.session_state.chat:
    if item["sender"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>{item['msg']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>{item['msg']}</div>", unsafe_allow_html=True)

# -----------------------
# USER INPUT
# -----------------------
user_input = st.text_input("Ask something:", value="", key="inputbox")

if st.button("Send"):
    if user_input.strip() != "":
        # Show user message
        st.session_state.chat.append({"sender": "user", "msg": user_input})

        # Show typing animation
        with st.spinner("🤖 Zeeshan ka Chatbot is typing..."):
            time.sleep(1)
            bot_reply = get_answer(user_input)

        # Add bot reply
        st.session_state.chat.append({"sender": "bot", "msg": bot_reply})

        st.session_state.inputbox = ""  # Clear input box
        st.rerun()
