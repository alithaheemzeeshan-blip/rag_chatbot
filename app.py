import streamlit as st
import pdfplumber
from openai import OpenAI

# -------------------- SETTINGS --------------------
st.set_page_config(page_title="Zeeshan ka Chatbot", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PDF_PATH = "data/Zeeshan_Chatbot_Company_Manual.pdf"

# -------------------- LOAD PDF --------------------
@st.cache_data
def load_pdf_text():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

pdf_text = load_pdf_text()

# -------------------- SIMPLE RAG RETRIEVAL --------------------
def retrieve_context(query):
    lines = pdf_text.split("\n")
    for line in lines:
        if query.lower() in line.lower():
            return line
    return "No direct match found in the PDF."

# -------------------- GET ANSWER --------------------
def get_answer(question):
    context = retrieve_context(question)

    system_prompt = f"""
You are Zeeshan ka Chatbot, a simple company helper bot.

If the PDF contains relevant information, use it.
If not, say:
"PDF mein iska jawab nahi mila, but here's a general answer:"

CONTEXT FROM PDF:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content  # FIXED


# -------------------- UI --------------------
st.title("🤖 Zeeshan ka Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

# INPUT BOX
user_input = st.text_input("Ask something:", key="input_q")

if st.button("Send"):
    if user_input.strip():
        answer = get_answer(user_input)

        st.session_state.chat.append(("You", user_input))
        st.session_state.chat.append(("Bot", answer))

        # Clear input box properly
        st.session_state.input_q = ""

# DISPLAY CHAT
st.write("---")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
