import streamlit as st
import pdfplumber
from openai import OpenAI

# -------------------- BASICS --------------------
st.set_page_config(page_title="J&Z ka Chatbot", layout="centered")

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

# -------------------- SIMPLE RETRIEVAL --------------------
def retrieve_context(query):
    lines = pdf_text.split("\n")
    for line in lines:
        if query.lower() in line.lower():
            return line
    return "No match found in PDF."

# -------------------- AI ANSWER --------------------
def get_answer(question):
    context = retrieve_context(question)

    system_prompt = f"""
You are Zeeshan ka Chatbot. Use the PDF context if relevant.

PDF CONTEXT:
{context}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    return res.choices[0].message.content


# -------------------- UI --------------------
st.title("🤖 Zeeshan ka Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------- INPUT FORM (NO ERROR!) --------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something:")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    answer = get_answer(user_input)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer))

# -------------------- DISPLAY CHAT --------------------
st.write("---")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖:** {msg}")


