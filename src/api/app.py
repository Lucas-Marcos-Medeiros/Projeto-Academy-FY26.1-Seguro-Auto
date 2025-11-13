import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Chat - Academy",
    page_icon="💬",
    layout="wide"
)

# Título da aplicação
st.title("💬 Chat - Academy")

# Inicializar o histórico de chat no session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens do histórico de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua mensagem aqui..."):
    # Adicionar mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Exibir mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Resposta do bot (simplesmente ecoa a mensagem do usuário)
    response = f"Você disse: {prompt}"
    
    # Adicionar resposta do bot ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Exibir resposta do bot
    with st.chat_message("assistant"):
        st.markdown(response)

# Botão para limpar o histórico
if st.button("🗑️ Limpar Conversa"):
    st.session_state.messages = []
    st.rerun()