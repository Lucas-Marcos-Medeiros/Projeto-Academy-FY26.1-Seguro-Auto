import streamlit as st
from genai.llm_client import llm
import os

# Carregar CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def chatbot_page():
    with open(os.path.join(os.path.dirname(__file__), "styles", "chatbot.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
    load_css()

    st.title(" Chatbot Assistente de Seguros")

    st.markdown("<div class='card'><h3>Converse com o assistente</h3></div>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for role, msg in st.session_state["messages"]:
        if role == "user":
            st.markdown(f"<div class='chat-bubble-user'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-assistant'>{msg}</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Digite sua mensagem:")

    if prompt:
        st.session_state["messages"].append(("user", prompt))

        reply = llm.invoke(prompt)
        resposta_texto = reply.content if hasattr(reply, "content") else reply

        st.session_state["messages"].append(("assistant", resposta_texto))

        st.rerun()
