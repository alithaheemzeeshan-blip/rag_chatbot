import streamlit as st
import openai
from streamlit_chat import message
from st_mic_recorder import mic_recorder
import time

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Voice ChatGPT", page_icon="🎤", layout="centered")

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------------- CSS ----------------
st.markdown("""
<style>
.chat-box {
    background-color: #202123;
    padding: 20px;
    border-radius: 15px;
    width: 100%;
    max-width: 750px;
    margin: auto;
}
.stTextInput>div>div>input {
    background: #3A3B3C;
    color: white;
}
.clear-button {
    background:#ff5555;
    padding:7px 15px;
    color:white;
    border-radius:8px;
    cursor:pointer;
}
</style>
""", unsafe_allow_html=True)

# ---------------- GPT FUNCTION ----------------
def ask_gpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "input_reset" not in st.session_state:
    st.session_state.input_reset = ""

# ---------------- TITLE + CLEAR BUTTON ----------------
st.title("🎤 Voice ChatGPT (Voice + Typing Animation)")

if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.session_state.last_input = ""
    st.rerun()

st.markdown('<div class="chat-box">', unsafe_allow_html=True)


# ---------------- SHOW CHAT HISTORY ----------------
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=(msg["role"] == "user"), key=f"msg_{i}")

# ---------------- TEXT INPUT ----------------
user_text = st.text_input("Ask me something:", value=st.session_state.get("input_reset", ""))
st.session_state.input_reset = ""


# ---------------- MICROPHONE INPUT ----------------
audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="🛑 Stop", just_once=True)

if audio:
    import speech_recognition as sr
    r = sr.Recognizer()

    with sr.AudioFile(audio['bytes']) as source:
        mic_audio = r.record(source)
        try:
            spoken_text = r.recognize_google(mic_audio)
        except:
            spoken_text = "I could not understand your voice."

    st.session_state.last_input = spoken_text
    st.session_state.messages.append({"role": "user", "content": spoken_text})

    # Typing animation placeholder
    with st.spinner("Assistant is typing…"):
        time.sleep(1)

    bot_reply = ask_gpt(spoken_text)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    st.rerun()


# ---------------- PROCESS TEXT INPUT ----------------
if user_text.strip() and user_text != st.session_state.last_input:

    st.session_state.last_input = user_text
    st.session_state.messages.append({"role": "user", "content": user_text})

    # ---- Typing animation ----
    with st.spinner("Assistant is typing…"):
        time.sleep(1)

    bot_reply = ask_gpt(user_text)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    st.session_state.input_reset = ""
    st.rerun()


st.markdown('</div>', unsafe_allow_html=True)
