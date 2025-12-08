import streamlit as st
import pdfplumber
from openai import OpenAI
import os

# -----------------------------
# Load API Key
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# -----------------------------
# Load PDF Data for RAG
# -----------------------------
def load_pdf_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except:
        text = ""
    return text

pdf_text = load_pdf_text("data/Zeeshan_Chatbot_Company_Manual.pdf")

# -----------------------------
# Generate Answer (RAG + AI)
# -----------------------------
def generate_answer(question):
    system_prompt = f"""
You are **Zeeshan ka Chatbot**, a corporate professional AI assistant.
Use this PDF knowledge when answering questions:

{pdf_text}

If the PDF does not contain the answer,
use your general AI knowledge but stay formal and helpful.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message["content"]

# -----------------------------
# Streamlit Interface
# -----------------------------

st.set_page_config(page_title="Zeeshan ka Chatbot", layout="wide")
st.title("🤖 Zeeshan ka Chatbot — Corporate AI Assistant")

st.write("Ask anything below. Your chatbot will use PDF + AI knowledge to answer!")

# Initialize session state variables
if "input_box" not in st.session_state:
    st.session_state.input_box = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------
# SAFE CLEAR INPUT FUNCTION
# -----------------------------
def clear_input():
    st.session_state.input_box = ""


# -----------------------------
# INPUT BOX + BUTTON
# -----------------------------
user_input = st.text_input(
    "Ask something:",
    key="input_box",
    placeholder="Type your question here..."
)

send = st.button("Send", on_click=clear_input)

if send and user_input:
    answer = generate_answer(user_input)
    st.session_state.chat_history.append({"q": user_input, "a": answer})


# -----------------------------
# DISPLAY CHAT BELOW
# -----------------------------
st.markdown("---")
st.subheader("💬 Conversation")

for msg in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {msg['q']}")
    st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg['a']}")
    st.markdown("---")
