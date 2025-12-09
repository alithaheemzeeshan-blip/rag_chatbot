import streamlit as st
import pdfplumber
from groq import Groq
from tavily import TavilyClient

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# YOUR REAL PDF LOCATION
PDF_PATH = "Zeeshan_Chatbot_Company_Manual.pdf"

# -------------------- LOAD PDF --------------------
@st.cache_data
def load_pdf_text():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            tx = page.extract_text()
            if tx:
                text += tx + "\n"
    return text

pdf_text = load_pdf_text()

# -------------------- RETRIEVAL --------------------
def retrieve_from_pdf(query):
    lines = pdf_text.split("\n")
    matches = [line for line in lines if query.lower() in line.lower()]
    return "\n".join(matches[:5]) if matches else "No match found in PDF."

# -------------------- WEB SEARCH --------------------
def search_web(query):
    result = tavily.search(query=query, max_results=3)
    if "results" not in result:
        return "No live data found."
    text = "\n".join([item["content"] for item in result["results"]])
    return text

# -------------------- AI ANSWER --------------------
def generate_answer(user_q):
    pdf_context = retrieve_from_pdf(user_q)
    web_context = search_web(user_q)

    system_prompt = f"""
You are Zeeshan ka Chatbot.
Use BOTH:
1. The PDF knowledge (company manual)
2. Live Internet updates (via Tavily)

PDF CONTEXT:
{pdf_context}

WEB CONTEXT:
{web_context}

Give a helpful answer.
"""

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_q}
        ]
    )

    return response.choices[0].message["content"]

# -------------------- UI --------------------
st.title("🤖 Zeeshan ka Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something:")
    send = st.form_submit_button("Send")

if send and user_input.strip():
    answer = generate_answer(user_input)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer))

# -------------------- DISPLAY CHAT --------------------
st.write("---")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
