import streamlit as st
import pdfplumber
from groq import Groq
from tavily import TavilyClient

# ---------------- CONFIG -------------------
st.set_page_config(page_title="Zeeshan RAG Chatbot", layout="centered")

GROQ_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

client = Groq(api_key=GROQ_KEY)
tavily = TavilyClient(api_key=TAVILY_KEY)

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"

# ---------------- LOAD PDF -------------------
@st.cache_data
def load_pdf():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

pdf_text = load_pdf()

# ---------------- SIMPLE PDF RETRIEVAL -----------
def retrieve_from_pdf(query):
    for line in pdf_text.split("\n"):
        if query.lower() in line.lower():
            return line
    return "No relevant information found in PDF."

# ---------------- WEB SEARCH ----------------------
def search_web(q):
    try:
        result = tavily.search(q)
        return result["results"][0]["content"]
    except:
        return "No live web data available."

# ---------------- LLM (GROQ) ----------------------
def generate_answer(user_q):
    pdf_info = retrieve_from_pdf(user_q)
    web_info = search_web(user_q)

    system_prompt = f"""
You are Zeeshan's intelligent RAG chatbot.
Use BOTH the PDF and live web search to answer
with accurate and updated information.

PDF DATA:
{pdf_info}

WEB DATA:
{web_info}

If PDF contradicts web, ALWAYS prefer web (latest info). 
Respond naturally, intelligently, and concisely.
"""

    response = client.chat.completions.create(
        model="llama3-8b",     # ✔ STABLE MODEL
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_q}
        ]
    )

    return response.choices[0].message["content"]


# ---------------- UI -------------------------------
st.title("🤖 Zeeshan's Smart RAG Chatbot (PDF + Web + LLM)")

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
