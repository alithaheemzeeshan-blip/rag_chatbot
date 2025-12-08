import streamlit as st
from openai import OpenAI
import pdfplumber
import os

st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

# ---------------------------
# Load PDF Data for RAG
# ---------------------------
def load_pdf_data(pdf_path="data/Zeeshan_Chatbot_Company_Manual.pdf"):
    if not os.path.exists(pdf_path):
        return ""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

RAG_DATA = load_pdf_data()

# ---------------------------
# OpenAI Client
# ---------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def generate_answer(user_q):
    """ Generates an answer using RAG + GPT """
    prompt = f"""
You are **Zeeshan ka Chatbot**, a professional AI assistant.
Use the RAG company manual below to answer questions accurately.

--- COMPANY MANUAL ---
{RAG_DATA}
----------------------

User Question: {user_q}

If the manual does NOT contain the answer, reply politely: 
"I'm sorry, this information is not in my company handbook."
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message["content"]


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🤖 Zeeshan ka Chatbot")

st.write("Ask anything related to your company policies or information.")

# Keep chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input box (WITHOUT setting session state directly)
user_input = st.text_input("Ask something:", key="input_box")

# When user presses Enter OR clicks Send
if st.button("Send"):
    if user_input.strip() != "":
        answer = generate_answer(user_input)

        # Save into history
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))

        # CLEAR INPUT BOX SAFELY
        st.session_state.input_box = ""  # safe reset via widget key

# ---------------------------
# DISPLAY CHAT MESSAGES
# ---------------------------

for role, msg in st.session_state.history:
    if role == "You":
        st.markdown(f"### 🧑‍💼 You:\n{msg}")
    else:
        st.markdown(f"### 🤖 Zeeshan ka Chatbot:\n{msg}")
