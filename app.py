import streamlit as st
from openai import OpenAI
import os
import pdfplumber

# -----------------------------
# OPENAI CLIENT
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# LOAD PDF DATA (RAG)
# -----------------------------
def load_pdf_text():
    folder = "data/"
    all_text = ""

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(folder, file)) as pdf:
                for page in pdf.pages:
                    all_text += page.extract_text() + "\n"

    return all_text if all_text else "No company documents found."

COMPANY_DATA = load_pdf_text()

# -----------------------------
# SYSTEM PROMPT (YOUR CHATBOT’S BRAIN)
# -----------------------------
SYSTEM_PROMPT = f"""
You are **Zeeshan ka Chatbot**, a professional corporate assistant and tech robot.

Your knowledge includes:
1. General AI knowledge (ChatGPT-level)
2. The company's internal policies, rules, and information found in this document:
---
{COMPANY_DATA}
---

Rules:
- Always reply professionally.
- If user asks about company things, answer using the PDF knowledge.
- If uncertain, politely say you don't have enough information.
"""

# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(user_query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )

    # NEW OPENAI FORMAT (fix for your error)
    return response.choices[0].message.content


# -----------------------------
# UI DESIGN
# -----------------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color:#00ffcc;'>
        🤖 Zeeshan ka Chatbot – Corporate Tech Assistant  
    </h1>
    <p style='text-align:center; color:#cccccc;'>
        Ask me anything. I can answer using AI + your PDF company data.
    </p>
    """,
    unsafe_allow_html=True,
)

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# CHAT INPUT
# -----------------------------
user_input = st.text_input("Ask something:", "")

# Auto-clear textbox after sending
if st.button("Send"):
    if user_input.strip() != "":
        st.session_state.history.append(("You", user_input))

        with st.spinner("Thinking... 🤖"):
            bot_reply = generate_answer(user_input)

        st.session_state.history.append(("ZeeshanBot", bot_reply))
        st.experimental_rerun()

# -----------------------------
# SHOW CHAT HISTORY WITH ANIMATION
# -----------------------------
for sender, message in st.session_state.history:
    if sender == "You":
        st.markdown(f"""
        <div style='text-align:right; padding:10px; margin:5px; background:#1e1e1e; border-radius:10px;'>
            <b>You:</b> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='animation: fadeIn 0.8s; padding:10px; margin:5px; background:#003333; border-radius:10px;'>
            <b>🤖 ZeeshanBot:</b> {message}
        </div>

        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0px); }}
        }}
        </style>
        """, unsafe_allow_html=True)
