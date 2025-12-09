import streamlit as st
import pdfplumber
from openai import OpenAI

# -------------------- APP SETTINGS --------------------
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

# -------------------- SIMPLE PDF RETRIEVAL --------------------
def retrieve_context(query):
    lines = pdf_text.split("\n")
    for line in lines:
        if query.lower() in line.lower():
            return line
    return ""  # empty means "no relevant company info found"

# -------------------- AI ANSWER --------------------
def get_answer(question):
    context = retrieve_context(question)

    system_prompt = f"""
You are **Zeeshan ka Chatbot**, a helpful assistant.

IMPORTANT RULES:
1. If the user asks **general knowledge questions** (politics, world events, technology, news, people, laws, etc.),
   → IGNORE the PDF completely and use your OWN updated AI knowledge.

2. ONLY use the PDF context for **company-related questions**, such as:
   - company rules
   - internal procedures
   - company values
   - anything directly related to the manual

3. If the PDF info is outdated or contradicts real-world facts:
   → ALWAYS prefer your up-to-date AI knowledge.

PDF CONTEXT (use ONLY when relevant to company topics):
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content


# -------------------- UI --------------------
st.title("🤖 Zeeshan ka Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------- INPUT FORM --------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something:")
    submitted = st.form_submit_button("Send")

# -------------------- PROCESS INPUT --------------------
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
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
