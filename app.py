import streamlit as st
import pdfplumber
import json
from groq import Groq
from tavily import TavilyClient

# ---------------------------- CONFIG ---------------------------------

st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

GROQ_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

client = Groq(api_key=GROQ_KEY)
tavily = TavilyClient(api_key=TAVILY_KEY)

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"

# ---------------------------- LOAD PDF ---------------------------------

@st.cache_data
def load_pdf_text():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

pdf_text = load_pdf_text()

# ---------------------------- SEARCH PDF (RAG) --------------------------

def search_pdf(query):
    lines = pdf_text.split("\n")
    matches = [line for line in lines if query.lower() in line.lower()]
    return "\n".join(matches[:5]) if matches else "No relevant PDF data found."

# ---------------------------- INTERNET SEARCH --------------------------

def search_internet(query):
    try:
        result = tavily.search(query=query, max_results=3)
        internet_text = "\n".join([item["content"] for item in result["results"]])
        return internet_text
    except Exception as e:
        return f"Internet search error: {e}"

# ---------------------------- GENERATE ANSWER --------------------------

def generate_answer(question):
    
    pdf_context = search_pdf(question)
    live_context = search_internet(question)

    system_prompt = f"""
You are Zeeshan ka Chatbot. You must use BOTH:
1. PDF RAG context
2. Internet latest data

Always combine both and give the most correct updated answer.

PDF CONTEXT:
{pdf_context}

INTERNET CONTEXT:
{live_context}
"""

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    # Correct Groq usage --> choices[0].message['content']
    return response.choices[0].message["content"]

# ---------------------------- UI --------------------------------------

st.title("🤖 Zeeshan ka Chatbot (RAG + LIVE Data)")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("chat_form", clear_on_submit=True):
    q = st.text_input("Ask something:")
    submit = st.form_submit_button("Send")

if submit and q.strip():
    answer = generate_answer(q)
    st.session_state.chat.append(("You", q))
    st.session_state.chat.append(("Bot", answer))

# ---------------------------- DISPLAY CHAT ------------------------------

st.write("---")

for sender, message in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {message}")
