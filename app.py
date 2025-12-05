import streamlit as st
from openai import OpenAI

# Load API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="AI Chatbot", layout="wide")

# --- Title ---
st.markdown("<h1 style='text-align:center;'>🤖 AI RAG Chatbot</h1>", unsafe_allow_html=True)
st.write("")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- CHAT DISPLAY ---
for msg in st.session_state["messages"]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.write(msg["content"])

# ============================================================
# 🎤 VOICE INPUT SECTION
# ============================================================

st.subheader("🎤 Speak your question:")

audio = st.audio_input("Click to record...")

user_voice_text = ""

if audio is not None:
    st.success("Audio recorded! Converting to text...")

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio.getvalue()
    )

    user_voice_text = transcript.text
    st.info(f"🗣️ You said: **{user_voice_text}**")

# ============================================================
# ⌨️ TEXT INPUT WITH AUTO-CLEAR
# ============================================================

if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

def send_text():
    st.session_state["submitted_text"] = st.session_state["text_input"]
    st.session_state["text_input"] = ""  # CLEAR after submit

user_text = st.text_input(
    "Ask me something:",
    key="text_input",
    on_change=send_text
)

user_text = st.session_state.get("submitted_text", "")

# ============================================================
# DETERMINE FINAL USER MESSAGE (VOICE > TEXT)
# ============================================================

final_user_message = ""

if user_voice_text:
    final_user_message = user_voice_text
elif user_text:
    final_user_message = user_text

# ============================================================
# PROCESS MESSAGE WITH OPENAI
# ============================================================

if final_user_message:
    # Add to chat history
    st.session_state["messages"].append({"role": "user", "content": final_user_message})

    with st.chat_message("user"):
        st.write(final_user_message)

    # Query AI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state["messages"]
    )

    bot_reply = response.choices[0].message["content"]

    # Save assistant reply
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.write(bot_reply)
