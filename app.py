import streamlit as st
from openai import OpenAI
import os
import pdfplumber

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", page_icon="🤖", layout="centered")

# -----------------------------
# API KEY
# -----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# LOAD PDF KNOWLEDGE
# -----------------------------
def load_pdf_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except:
        pass
    return text

PDF_TEXT = load_pdf_text("Zeeshan_Chatbot_Company_Manual.pdf")

# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_box" not in st.session_state:
    st.session_state.input_box = ""

# -----------------------------
# RAG MODEL
# -----------------------------
def generate_answer(question):
    full_context = f"""
    Company Manual Knowledge:
    {PDF_TEXT}

    User Question:
    {question}

    Generate the best answer using the company manual first. 
    If answer is missing, use general AI knowledge.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Zeeshan ka Chatbot, a smart assistant."},
            {"role": "user", "content": full_context}
        ]
    )

    return response.choices[0].message["content"]

# -----------------------------
# UI TITLE
# -----------------------------
st.title("🤖 Zeeshan ka Chatbot")
st.write("Ask anything! I will use company PDF + AI knowledge.")

# -----------------------------
# USER INPUT BAR
# -----------------------------
user_input = st.text_input(
    "Ask something:",
    value=st.session_state.input_box,
    key="input_bar"
)

# -----------------------------
# PROCESS QUESTION
# -----------------------------
if st.button("Send"):
    if user_input.strip() != "":
        answer = generate_answer(user_input)

        st.session_state.chat_history.append({"q": user_input, "a": answer})

        # CLEAR TEXT BOX
        st.session_state.input_box = ""
        st.rerun()

# -----------------------------
# SHOW CHAT HISTORY
# -----------------------------
st.markdown("### 💬 Conversation")

for msg in reversed(st.session_state.chat_history):
    st.write(f"🧑 **You:** {msg['q']}")
    st.write(f"🤖 **Zeeshan ka Chatbot:** {msg['a']}")
    st.markdown("---")
