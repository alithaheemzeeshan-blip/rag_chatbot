import streamlit as st
import pdfplumber
from openai import OpenAI

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"

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

# -------------------- SIMPLE CONTEXT SEARCH --------------------
def retrieve_context(query):
    lines = pdf_text.split("\n")
    for line in lines:
        if query.lower() in line.lower():
            return line
    return "No relevant line found in PDF."

# -------------------- AI ANSWER --------------------
def get_answer(question):
    context = retrieve_context(question)

    system_prompt = f"""
You are Zeeshan ka Chatbot. You ALWAYS answer using this PDF context if helpful:

PDF CONTEXT:
{context}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )

    return res.choices[0].message.content


# -------------------- CUSTOM STYLING --------------------
st.markdown("""
<style>

.chat-container {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 10px;
}

.user-bubble {
    background-color: #4CAF50;
    color: white;
    padding: 12px 18px;
    margin: 6px;
    border-radius: 10px;
    max-width: 75%;
    align-self: flex-end;
}

.bot-bubble {
    background-color: #2C2C2C;
    color: #FFD369;
    padding: 12px 18px;
    margin: 6px;
    border-radius: 10px;
    max-width: 75%;
    align-self: flex-start;
    border: 1px solid #444;
}

.chat-wrapper {
    display: flex;
    flex-direction: column;
}

.input-box input {
    background-color: #2b2b2b !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

.send-btn {
    background-color: #FFD369 !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)



# -------------------- UI --------------------
st.markdown("<h1 style='text-align:center; color:#FFD369;'>🤖 Zeeshan ka Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#BBBBBB;'>Simple RAG | Custom UI | PDF-Powered</p>", unsafe_allow_html=True)

if "chat" not in st.session_state:
    st.session_state.chat = []

# FORM (SAFE)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something:", key="msg", placeholder="Type your question here...")
    submitted = st.form_submit_button("Send", help="Ask Zeeshan's Chatbot", type="primary")

# PROCESS SUBMISSION
if submitted and user_input.strip():
    answer = get_answer(user_input)

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer))

# -------------------- DISPLAY CHAT --------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"<div class='chat-wrapper'><div class='user-bubble'>🧑 <b>You:</b><br>{msg}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-wrapper'><div class='bot-bubble'>🤖 <b>Zeeshan ka Chatbot:</b><br>{msg}</div></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("<br><p style='text-align:center; color:#555;'>Made with ❤️ for Zeeshan</p>", unsafe_allow_html=True)
