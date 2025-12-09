import streamlit as st
import pdfplumber
import os
from tavily import TavilyClient
from groq import Groq

# ---------------- BASIC CONFIG -------------------
st.set_page_config(page_title="Zeeshan RAG Chatbot", layout="centered")

GROQ_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

client = Groq(api_key=GROQ_KEY)
tavily = TavilyClient(api_key=TAVILY_KEY)

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"

# ---------------- LOAD PDF ------------------------
@st.cache_data
def load_pdf():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
    return text

pdf_text = load_pdf()

# ---------------- SIMPLE RAG -----------------------
def retrieve_from_pdf(query):
    best = ""
    for line in pdf_text.split("\n"):
        if query.lower() in line.lower():
            best = line
            break
    return best if best else "NO MATCH FOUND IN PDF"

# ---------------- WEB SEARCH -----------------------
def web_search(q):
    try:
        result = tavily.search(q)
        return result["results"][0]["content"]
    except:
        return "Web search unavailable."

# ---------------- GROQ LLM -------------------------
def generate_answer(query):
    
    context_pdf = retrieve_from_pdf(query)
    context_web = web_search(query)

    system_msg = f"""
You are Zeeshan's RAG chatbot.
Use BOTH the PDF and live web info.

PDF DATA:
{context_pdf}

WEB DATA:
{context_web}

Always combine both sources.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message["content"]

# ---------------- UI -------------------------------
st.title("🤖 Zeeshan RAG Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("ask_form", clear_on_submit=True"):
    q = st.text_input("Ask something:")
    send = st.form_submit_button("Send")

if send and q.strip():
    ans = generate_answer(q)
    st.session_state.chat.append(("You", q))
    st.session_state.chat.append(("Bot", ans))

st.write("---")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")
