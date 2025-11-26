import streamlit as st
import sys
import os

# -------------------------------------------------------
# 1. Ajusta o sys.path ANTES de importar qualquer coisa
# -------------------------------------------------------
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

# ------------------------------
# CONFIGURAÇÃO DA APLICAÇÃO (PRIMEIRO)
# ------------------------------
st.set_page_config(
    page_title="SeguraBOT", 
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import após config
from src.genai.llm_client import llm
from src.app.calculator import calcular_premio
from src.app.data_manager import get_data_manager
from src.genai.llm_context import get_context_enricher
import time

# ------------------------------
# LOADING DE DADOS - BEM VISÍVEL
# ------------------------------
@st.cache_resource(show_spinner=False)
def init_data_manager():
    """Inicializa o gerenciador de dados"""
    dm = get_data_manager()
    return dm

# Verifica se já carregou
if 'data_loaded' not in st.session_state:
    # Container no topo para loading
    loading_container = st.container()
    
    with loading_container:
        st.markdown("### 🔄 Carregando Sistema")
        st.info("**Por favor, aguarde enquanto carregamos os dados...**")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Inicializando...")
            progress_bar.progress(20)
            
            status_text.text("Conectando ao dataset...")
            progress_bar.progress(40)
            
            data_manager = init_data_manager()
            
            status_text.text("Carregando tabelas...")
            progress_bar.progress(80)
            
            st.session_state['data_manager'] = data_manager
            st.session_state['data_loaded'] = True
            
            progress_bar.progress(100)
            status_text.text("Concluído!")
            
            time.sleep(1)
            
            # Limpa e mostra sucesso
            loading_container.empty()
            success_msg = st.success("✅ **Sistema carregado com sucesso!**")
            time.sleep(1.5)
            success_msg.empty()
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ **Erro ao carregar dados:** {e}")
            st.info("💡 Tente recarregar a página (F5)")
            st.stop()

# Recupera data_manager
data_manager = st.session_state.get('data_manager')

# Inicializa sessão
if "page" not in st.session_state:
    st.session_state["page"] = "chat"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ------------------------------
# FUNÇÃO PARA TELA DE CHAT
# ------------------------------
def chat_page():
    st.title("💬 Assistente Inteligente de Seguros")
    st.markdown("Tire suas dúvidas sobre seguros automotivos com nosso assistente baseado em IA")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Menu")
        
        if st.button("🔢 Calculadora de Prêmio", use_container_width=True):
            st.session_state["page"] = "calculadora"
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("## 💡 Como usar")
        st.info("""
        **Exemplos de perguntas:**
        - Quanto custa um seguro para meu carro?
        - Quais fatores influenciam o preço?
        - Como funciona o cálculo do prêmio?
        - Qual importância de se ter um seguro?
        """)
        
        st.markdown("---")
        
        with st.expander("📊 Sobre os Dados"):
            if data_manager:
                data_summary = data_manager.get_all_tables_summary()
                st.markdown(data_summary)
        
        st.markdown("---")
        
        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
        
        st.markdown("---")
        st.caption("Powered by IA")

    # Mensagem de boas-vindas
    if not st.session_state["messages"]:
        st.info("""
        👋 **Olá! Sou seu assistente de seguros automotivos.**
        
        Posso ajudar você com:
        - ✅ Informações sobre seguros de veículos
        - ✅ Fatores que influenciam o preço
        - ✅ Comparações entre regiões e modelos
        - ✅ Dúvidas sobre cobertura e cálculos
        
        **Como posso ajudar você hoje?**
        """)
    
    # Container para mensagens
    chat_container = st.container()
    
    with chat_container:
        for role, content in st.session_state["messages"]:
            with st.chat_message(role):
                # Mostra conteúdo SEM limpeza (versão original)
                st.markdown(content)

    # Input do usuário
    prompt = st.chat_input("Digite sua pergunta aqui...")
    
    if prompt:
        st.session_state["messages"].append(("user", prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando sua pergunta..."):
                try:
                    enricher = get_context_enricher()
                    prompt_enriquecido = enricher.enrich_prompt(prompt, st.session_state["messages"])
                    
                    resposta = llm.invoke(prompt_enriquecido)
                    resposta_texto = resposta.content if hasattr(resposta, "content") else str(resposta)
                    
                    # Mostra resposta SEM limpeza (versão original)
                    st.markdown(resposta_texto)
                
                except Exception as e:
                    resposta_texto = "Desculpe, ocorreu um erro. Tente novamente."
                    st.error(resposta_texto)
                    print(f"Erro: {e}")

        st.session_state["messages"].append(("assistant", resposta_texto))
        st.rerun()


# ------------------------------
# PÁGINA DE BOAS-VINDAS
# ------------------------------
def welcome_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🚗 Sistema de Seguros")
        st.markdown("### Inteligência Artificial para Seguros Automotivos")
    
    st.markdown("---")
    st.markdown("## ✨ O que você pode fazer aqui?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💬 Assistente Inteligente
        
        Converse com nosso assistente baseado em IA para:
        - 🔍 Tirar dúvidas sobre seguros
        - 📊 Consultar estatísticas e tendências
        - 💡 Obter recomendações personalizadas
        - 📈 Entender fatores de risco
        """)
        
        if st.button("💬 Iniciar Conversa", use_container_width=True, type="primary"):
            st.session_state["page"] = "chat"
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 🧮 Calculadora de Prêmio
        
        Calcule o valor estimado do seu seguro:
        - 🚗 Escolha modelo e ano do veículo
        - 👤 Informe seu perfil de condutor
        - 📍 Selecione sua região
        - 💰 Receba cotação personalizada
        """)
        
        if st.button("🧮 Calcular Prêmio", use_container_width=True):
            st.session_state["page"] = "calculadora"
            st.rerun()
    
    st.markdown("---")
    st.info("""
    💡 **Dica:** Comece conversando com o assistente para entender melhor como funciona o cálculo de seguros!
    """)


# ------------------------------
# ROTAS
# ------------------------------
if st.session_state["page"] == "welcome":
    welcome_page()
elif st.session_state["page"] == "chat":
    chat_page()
elif st.session_state["page"] == "calculadora":
    calcular_premio()