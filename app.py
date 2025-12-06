# =========================================
#     Z E E S H A N   K A   C H A T B O T
#     Professional RAG AI Assistant
# =========================================

import streamlit as st
import time
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# ------------------------------
# STREAMLIT CONFIG + UI DESIGN
# ------------------------------
st.set_page_config(
    page_title="Zeeshan Ka Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------- Custom CSS (Animations + UI) ----------
st.markdown("""
<style>

    body {
        background-color: #0e1117 !important;
    }

    .main {
        background-color: #0e1117;
        color: white;
    }

    .title_text {
        font-size: 42px;
        font-weight: bold;
        color: #3b82f6;
        text-align: center;
        margin-bottom: 20px;
    }

    .bot_msg {
        background-color: #111827;
        padding: 14px;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        width: fit-content;
        margin-bottom: 10px;

        /* Animation */
        animation: fadeIn 0.6s ease-in-out;
    }

    .user_msg {
        background-color: #1f2937;
        padding: 14px;
        border-radius: 12px;
        width: fit-content;
        margin-bottom: 10px;
        margin-left: auto;

        animation: fadeIn 0.6s ease-in-out;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0px); }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        color: white;
        padding: 20px;
    }

    .sidebar_title {
        font-size: 22px;
        font-weight: bold;
        color: #3b82f6;
        margin-bottom: 20px;
    }

</style>
""", unsafe_allow_html=True)

# -------------------
# HEADER TITLE
# -------------------
st.markdown("<div class='title_text'>🤖 Zeeshan Ka Chatbot — Corporate AI Assistant</div>",
            unsafe_allow_html=True)

# ------------------------------
# SIDEBAR BRANDING
# ------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar_title'>Zeeshan AI Assistant</div>", unsafe_allow_html=True)
    st.image("https://i.postimg.cc/6qk8njLC/robot-tech.png", use_column_width=True)
    st.write("This chatbot uses RAG + AI to answer company-related questions.")
    st.write("Powered by OpenAI + PDF Knowledge Base.")


# ------------------------------
# LOAD PDF KNOWLEDGE (RAG)
# ------------------------------
def load_pdf_text():
    folder = "data"
    text = ""

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf = PdfReader(os.path.join(folder, file))
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    return text


knowledge_base = load_pdf_text()


# ------------------------------
# AI CLIENT
# ------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # OR st.secrets["OPENAI_API_KEY"]


# ------------------------------
# RAG + ChatGPT Response
# ------------------------------
def generate_answer(question):
    prompt = f"""
You are Zeeshan's Corporate AI Assistant.
Answer using the policy and information below.

--- COMPANY MANUAL ---
{knowledge_base}
-----------------------

USER QUESTION:
{question}

Give a helpful, professional answer.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message["content"]


# ------------------------------
# SESSION STATE (Chat History)
# ------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []


# ------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"<div class='user_msg'>🧑‍💼 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot_msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)


# ------------------------------
# TYPING ANIMATION FUNCTION
# ------------------------------
def animated_text(full_text):
    output = st.empty()
    text = ""

    for char in full_text:
        text += char
        output.markdown(f"<div class='bot_msg'>🤖 {text}</div>", unsafe_allow_html=True)
        time.sleep(0.015)  # typing speed


# ------------------------------
# INPUT BOX
# ------------------------------
user_input = st.text_input("Ask something:", key="user_input", placeholder="Type your question here…")

if user_input:
    # Add user message
    st.session_state.chat.append({"role": "user", "content": user_input})

    # Generate AI response
    answer = generate_answer(user_input)

    # Add bot response
    st.session_state.chat.append({"role": "assistant", "content": answer})

    # Clear input
    st.session_state.user_input = ""

    # Rerun to display
    st.rerun()

