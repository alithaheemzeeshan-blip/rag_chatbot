import streamlit as st
import pdfplumber
from openai import OpenAI

# -------------------- PAGE SETUP --------------------
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

# -------------------- IMPROVED SIMPLE RAG --------------------
def retrieve_context(query):
    lines = pdf_text.split("\n")
    query_words = query.lower().split()

    scored = []

    for line in lines:
        score = 0
        lower_line = line.lower()
        for word in query_words:
            if word in lower_line:
                score += 1
        if score > 0:
            scored.append((score, line))

    if not scored:
        return "No matching info found in PDF."

    scored.sort(reverse=True)
    best_lines = [line for score, line in scored[:7]]  # top 7 lines
    return "\n".join(best_lines)


# -------------------- AI ANSWER --------------------
def get_answer(question):
    context = retrieve_context(question)

    system_prompt = f"""
You are **Zeeshan ka Chatbot**, a friendly and smart assistant.

Your rules:

1. ALWAYS use the PDF context FIRST.
2. If the PDF is missing details or outdated, ALSO use general AI knowledge.
3. Your final answer must be a combination of:
   - Information found in the PDF
   - Extra AI knowledge (latest information)
4. Do NOT invent things that conflict with the PDF.
5. If the PDF is outdated (like 2023), expand the answer with updated info.

PDF CONTEXT (may be partial or old):
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message["content"]


# -------------------- UI --------------------
st.title("🤖 Zeeshan ka Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Input form (clears automatically)
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
        st.markdown(f"**🤖 Zeeshan ka Chatbot:** {msg}")
