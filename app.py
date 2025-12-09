import streamlit as st
from groq import Groq
from tavily import TavilyClient
import pdfplumber

# --------------------------
# PAGE SETTINGS
# --------------------------
st.set_page_config(page_title="Zeeshan ka Smart RAG Chatbot", layout="centered")
st.title("🤖 Zeeshan ka Smart RAG Chatbot")

# --------------------------
# LOAD API KEYS
# --------------------------
GROQ_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

client = Groq(api_key=GROQ_KEY)
tavily = TavilyClient(api_key=TAVILY_KEY)

PDF_PATH = "data/manual.pdf"


# --------------------------
# LOAD PDF
# --------------------------
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


# --------------------------
# BASIC RETRIEVAL
# --------------------------
def retrieve_from_pdf(query):
    for line in pdf_text.split("\n"):
        if query.lower() in line.lower():
            return line
    return "No relevant PDF data found."


# --------------------------
# INTERNET SEARCH (TAVILY)
# --------------------------
def web_search(q):
    try:
        res = tavily.search(query=q, max_results=4)
        return "\n".join([r["content"] for r in res["results"]])
    except:
        return "Internet search unavailable."


# --------------------------
# GENERATE ANSWER (GROQ)
# --------------------------
def generate_answer(user_q):
    pdf_context = retrieve_from_pdf(user_q)
    web_context = web_search(user_q)

    prompt = f"""
Use ALL of the following sources to answer:

📘 PDF Knowledge:
{pdf_context}

🌐 Internet Search:
{web_context}

💡 If PDF or internet do not have the answer, use your own intelligence.

User Question: {user_q}
"""

    response = client.chat.completions.create(
        model="llama3-8b",     # ✔ WORKING MODEL
        messages=[
            {"role": "system", "content": "You are Zeeshan's RAG chatbot. Be accurate, friendly, updated."},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message["content"]


# --------------------------
# CHAT UI
# --------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []


with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask anything:")
    send = st.form_submit_button("Send")

if send and user_input.strip():
    answer = generate_answer(user_input)

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer))


# --------------------------
# DISPLAY CHAT MESSAGES
# --------------------------
st.write("---")
for sender, message in st.session_state.chat:
    if sender == "You":
        st.markdown(f"🧑 **You:** {message}")
    else:
        st.markdown(f"🤖 **Bot:** {message}")
