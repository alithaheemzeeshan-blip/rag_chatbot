import streamlit as st
import pdfplumber
import os
from tavily import TavilyClient
from groq import Groq

# ---------------- CONFIG -------------------
st.set_page_config(page_title="Zeeshan RAG Chatbot", layout="centered")

# Load API keys
GROQ_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

client = Groq(api_key=GROQ_KEY)
tavily = TavilyClient(api_key=TAVILY_KEY)

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"

# --------------- LOAD PDF -------------------
@st.cache_data
def load_pdf():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for pg in pdf.pages:
            t = pg.extract_text()
            if t:
                text += t + "\n"
    return text

pdf_text = load_pdf()

# --------------- SIMPLE PDF SEARCH ----------
def retrieve_from_pdf(query):
    for line in pdf_text.split("\n"):
        if query.lower() in line.lower():
            return line
    return "No relevant info found inside PDF."

# --------------- WEB SEARCH -----------------
def web_search(query):
    try:
        result = tavily.search(query)
        return result["results"][0]["content"]
    except Exception:
        return "Web search unavailable."

# --------------- GROQ LLM -------------------
def generate_answer(query):
    pdf_info = retrieve_from_pdf(query)
    web_info = web_search(query)

    system_prompt = f"""
You are Zeeshan's smart RAG chatbot.
Use BOTH the company PDF and live internet search.

PDF INFORMATION:
{pdf_info}

WEB INFORMATION:
{web_info}

Always answer based on both sources.
If PDF has no match, rely on web.
"""

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # ✔ SAFE MODEL
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message["content"]

# ---------------- UI ------------------------
st.title("🤖 Zeeshan RAG Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("ask_form", clear_on_submit=True):
    question = st.text_input("Ask something:")
    send = st.form_submit_button("Send")

if send and question.strip():
    answer = generate_answer(question)
    st.session_state.chat.append(("You", question))
    st.session_state.chat.append(("Bot", answer))

st.write("---")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")
