import streamlit as st
import time
import os
from openai import OpenAI
from PyPDF2 import PdfReader

# ---------------------------
# 1. PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Zeeshan ka Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# 2. OPENAI CLIENT
# ---------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------
# 3. LOAD PDF FOR RAG
# ---------------------------
@st.cache_data
def load_pdf():
    pdf_path = "data/Zeeshan_Chatbot_Company_Manual.pdf"
    if not os.path.exists(pdf_path):
        return "No company policy PDF found."

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text

company_knowledge = load_pdf()

# ---------------------------
# 4. GENERATE AI ANSWER (RAG)
# ---------------------------
def generate_answer(question):
    system_prompt = f"""
You are **Zeeshan ka Chatbot**, a helpful, corporate-style AI assistant.

You MUST use the knowledge below when answering:

--- COMPANY KNOWLEDGE ---
{company_knowledge}
--------------------------

If the user's question is related to company operations, policies, rules, or details,
answer using the PDF knowledge. If not found, answer like a normal AI but in a 
professional friendly tone.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message["content"]


# ---------------------------
# 5. TYPING ANIMATION
# ---------------------------
def render_typing(text):
    placeholder = st.empty()
    typing = ""
    for char in text:
        typing += char
        placeholder.markdown(f"🟦 **Zeeshan ka Chatbot:** {typing}")
        time.sleep(0.01)


# ---------------------------
# 6. CHAT HISTORY SESSION
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------
# 7. UI TITLE
# ---------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#00BFFF;'>
        🤖 Zeeshan ka Chatbot
    </h1>
    <p style='text-align:center; font-size:18px; color:#CCCCCC;'>
        Your professional assistant powered by AI + Company Knowledge
    </p>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ---------------------------
# 8. DISPLAY CHAT HISTORY
# ---------------------------
for role, msg in st.session_state.history:
    if role == "You":
        st.markdown(f"🟨 **You:** {msg}")
    else:
        st.markdown(f"🟦 **Zeeshan ka Chatbot:** {msg}")

# ---------------------------
# 9. USER INPUT
# ---------------------------
user_input = st.text_input("Ask something:", "", key="chatbox")

if st.button("Send"):
    if user_input.strip() != "":
        st.session_state.history.append(("You", user_input))

        with st.spinner("Zeeshan ka Chatbot is thinking..."):
            bot_reply = generate_answer(user_input)

        st.session_state.history.append(("Zeeshan ka Chatbot", bot_reply))

        st.rerun()  # CLEAR TEXTBOX + RELOAD CHAT


# ---------------------------
# 10. FOOTER
# ---------------------------
st.write("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Made for Zeeshan • AI Powered RAG Chatbot</p>",
    unsafe_allow_html=True
)
