import streamlit as st
import pandas as pd
import os
import sys
import re

# Ajusta path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

# Importações internas
from src.genai.llm_client import llm
from src.app.data_manager import get_data_manager
from src.genai.llm_context import get_context_enricher
from src.analises.auxiliary_data_analyzer import get_auxiliary_analyzer


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def clean_llm_response(text) -> str:
    """
    Remove formatação problemática da LLM
    VERSÃO MELHORADA
    """
    # Se for dict, tenta extrair o texto
    if isinstance(text, dict):
        text = text.get('content') or text.get('text') or str(text)
    
    # Se não for string, converte
    if not isinstance(text, str):
        text = str(text)
    
    if not text or text == 'None':
        return ""

    # Remove markdown
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("###", "")
    text = text.replace("##", "")
    text = text.replace("#", "")
    
    # Remove asteriscos
    text = re.sub(r'\*+', '', text)
    
    # Remove caracteres de escape do LaTeX
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove caracteres especiais problemáticos
    text = text.replace('~', '')
    text = text.replace('^', '')
    text = text.replace('∗', '')
    text = text.replace('`', '')
    
    # Corrige caracteres acentuados mal formatados
    replacements = {
        'aˊ': 'á',
        'eˊ': 'é',
        'ıˊ': 'í',
        'oˊ': 'ó',
        'uˊ': 'ú',
        'a~': 'ã',
        'o~': 'õ',
        'c\\': 'ç',
        'pre^': 'prê',
        'e^': 'ê',
        'o^': 'ô',
        'veıˊ': 'veí',
        'baˊ': 'bá',
        'poˊ': 'pó',
        'paıˊ': 'paí',
        'seˊ': 'sé',
        'ca~': 'cã',
        'ªa': 'a',
        'ºo': 'o'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    
    # Corrige pontuação
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'([,.;:!?])([A-Za-zÀ-ÿ])', r'\1 \2', text)
    
    # Corrige formato de moeda
    text = re.sub(r'R\s*\$?\s*(\d)', r'R$ \1', text)
    text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)
    
    # Primeiro, normaliza espaços após pontos
    text = re.sub(r'\.\s+([A-Z])', r'. \1', text)
    
    # Adiciona quebras entre parágrafos apenas:
    # - Quando há palavras-chave que iniciam novo parágrafo
    # - Ou quando há uma sequência de frases longas que indica mudança de assunto
    
    # Palavras-chave que iniciam novos parágrafos
    paragraph_starters = [
        'Diversos fatores',
        'Comparando',
        'Como recomendação',
        'Por fim',
        'Além disso',
        'Em relação',
        'Quanto',
        'Vale ressaltar',
        'É importante',
        'Neste contexto',
        'Outro ponto',
        'Adicionalmente'
    ]
    
    for starter in paragraph_starters:
        text = text.replace(f'. {starter}', f'.\n\n{starter}')

    return text.strip()


def get_combined_casco_data_local(data_manager):
    """Combina dados dos dois semestres"""
    df1 = data_manager.get_table("casco_sem1").copy()
    df2 = data_manager.get_table("casco_sem2").copy()

    df1["semestre"] = 1
    df1["periodo"] = "1º Semestre 2019"

    df2["semestre"] = 2
    df2["periodo"] = "2º Semestre 2019"

    combined = pd.concat([df1, df2], ignore_index=True)
    return combined


def format_currency(value):
    """Formata valor como moeda brasileira"""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def get_risk_level(frequencia, severidade):
    """
    Determina nível de risco baseado em frequência e severidade
    """
    # Se ambos forem zero, não há dados suficientes
    if frequencia == 0 and severidade == 0:
        return None, None
    
    score = (frequencia * 1000) + (severidade / 10000)
    
    if score < 0.5:
        return "🟢 Baixo", "success"
    elif score < 2.0:
        return "🟡 Médio", "warning"
    else:
        return "🔴 Alto", "error"


def get_comparison_stats(df, modelo, ano):
    """
    Calcula estatísticas comparativas para o modelo
    """
    stats = {}
    
    # Filtra por modelo
    df_modelo = df[df["modelo"] == modelo]
    
    if not df_modelo.empty:
        stats["premio_medio_modelo"] = df_modelo["premio1"].mean()
        stats["premio_min_modelo"] = df_modelo["premio1"].min()
        stats["premio_max_modelo"] = df_modelo["premio1"].max()
        stats["total_registros"] = len(df_modelo)
    
    # Filtra por ano próximo (±2 anos)
    df_ano_similar = df[(df["ano"] >= ano - 2) & (df["ano"] <= ano + 2)]
    
    if not df_ano_similar.empty:
        stats["premio_medio_ano"] = df_ano_similar["premio1"].mean()
    
    return stats


# ============================================================
# FUNÇÃO DE CÁLCULO ATUARIAL
# ============================================================

def calcular_premio_atuarial(modelo, ano, sexo, regiao_desc, faixa_desc):
    """Cálculo principal do prêmio"""

    data_manager = get_data_manager()
    enricher = get_context_enricher()
    aux_analyzer = get_auxiliary_analyzer()

    df = get_combined_casco_data_local(data_manager)

    # Filtro por modelo
    df_modelo = df[df["modelo"] == modelo]
    if df_modelo.empty:
        return {"erro": True, "mensagem": f"Modelo '{modelo}' não encontrado em nossa base de dados."}

    # Filtro por faixa
    df_faixa = df_modelo[df_modelo["faixa_desc"] == faixa_desc]
    if df_faixa.empty:
        df_faixa = df_modelo.copy()

    # Filtro por região
    df_regiao = df_faixa[df_faixa["regiao_desc"] == regiao_desc]
    if df_regiao.empty:
        df_regiao = df_faixa.copy()

    registro = df_regiao.iloc[0].to_dict()
    premio_hist = registro.get("premio1", 0)

    # Frequência e indenização
    freq_cols = [c for c in df.columns if "freq_sin" in c]
    inden_cols = [c for c in df.columns if "indeniz" in c]

    freq_total = sum(registro.get(c, 0) for c in freq_cols)
    inden_total = sum(registro.get(c, 0) for c in inden_cols)

    frequencia_media = freq_total / len(freq_cols) if freq_cols else 0
    severidade_media = inden_total / len(inden_cols) if inden_cols else 0

    premio_estimado = premio_hist

    if frequencia_media > 0 and severidade_media > 0:
        premio_estimado = frequencia_media * severidade_media

    # Ajustes
    fator_idade = max(0.7, min(1.2, (2025 - ano) * 0.01 + 0.9))
    premio_estimado *= fator_idade

    if sexo == "M":
        premio_estimado *= 1.10
    elif sexo == "F":
        premio_estimado *= 0.97

    # Ajuste por estado
    if "SP" in regiao_desc:
        premio_estimado *= 1.15
    elif "RJ" in regiao_desc:
        premio_estimado *= 1.22

    # Estatísticas comparativas
    stats_comparativas = get_comparison_stats(df, modelo, ano)

    # Contexto adicional
    contexto_adicional = None
    try:
        contexto_adicional = enricher.get_calculator_context(
            modelo, ano, sexo, regiao_desc, faixa_desc
        )
    except Exception as e:
        print(f"Aviso: Não foi possível gerar contexto adicional: {e}")

    # Perfil de risco
    uf_map = {
        "São Paulo": "SP", "Rio de Janeiro": "RJ", "Minas Gerais": "MG",
        "Paraná": "PR", "Santa Catarina": "SC", "Rio Grande do Sul": "RS",
        "Pernambuco": "PE", "Bahia": "BA", "Ceará": "CE"
    }

    uf = next((sig for est, sig in uf_map.items() if est in regiao_desc), None)

    perfil_risco = None
    if uf:
        try:
            perfil_risco = aux_analyzer.get_integrated_risk_profile(modelo, uf)
        except Exception as e:
            print(f"Aviso: Não foi possível gerar perfil de risco: {e}")

    return {
        "erro": False,
        "modelo": modelo,
        "ano": ano,
        "sexo": sexo,
        "regiao": regiao_desc,
        "faixa": faixa_desc,
        "premio_estimado": round(premio_estimado, 2),
        "premio_historico": round(premio_hist, 2),
        "frequencia": round(frequencia_media, 6),
        "severidade": round(severidade_media, 2),
        "registro_utilizado": df_regiao,
        "contexto_adicional": contexto_adicional,
        "stats_comparativas": stats_comparativas,
        "periodo_dados": registro.get("periodo", "Não especificado"),
        "perfil_risco": perfil_risco
    }


# ============================================================
# INTERFACE STREAMLIT - VERSÃO FINAL
# ============================================================

def calcular_premio():
    # Header
    st.title("🔢 Calculadora de Seguro Automotivo")
    st.markdown("---")
    
    # Botão voltar no topo
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Voltar", use_container_width=True):
            st.session_state["page"] = "chat"
            st.rerun()
    
    # Carrega dados
    data_manager = get_data_manager()
    
    with st.spinner("Carregando dados..."):
        df = get_combined_casco_data_local(data_manager)
    
    if df.empty:
        st.error("❌ Não foi possível carregar os dados. Por favor, tente novamente.")
        return

    # Informação sobre os dados
    with st.expander("ℹ️ Sobre os Dados"):
        st.info(f"""
        📊 **Base de dados:** {len(df):,} registros de seguros automotivos
        
        📅 **Período:** 1º e 2º Semestres de 2019
        
        🎯 **Cobertura:** Dados históricos reais de apólices de seguro
        """)

    st.markdown("### 📋 Preencha as Informações do Veículo")
    
    # Formulário
    with st.form("form_calculo", clear_on_submit=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            modelos = sorted(df["modelo"].dropna().unique())
            modelo = st.selectbox(
                "🚗 Modelo do Veículo",
                modelos,
                help="Selecione o modelo do seu veículo"
            )
            
            anos = sorted(df["ano"].dropna().astype(int).unique(), reverse=True)
            ano = st.selectbox(
                "📅 Ano do Veículo",
                anos,
                help="Ano de fabricação"
            )
            
            sexos = ["M", "F"]
            sexo_label = st.selectbox(
                "👤 Sexo do Condutor Principal",
                ["Masculino", "Feminino"],
                help="Sexo do principal condutor"
            )
            sexo = "M" if sexo_label == "Masculino" else "F"
        
        with col2:
            faixas = sorted(df["faixa_desc"].dropna().unique())
            faixa_desc = st.selectbox(
                "🎂 Faixa Etária",
                faixas,
                help="Faixa etária do condutor principal"
            )
            
            regioes = sorted(df["regiao_desc"].dropna().unique())
            regiao_desc = st.selectbox(
                "📍 Região",
                regioes,
                help="Região onde o veículo circula"
            )
        
        st.markdown("---")
        
        # Botão de calcular centralizado e destacado
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "Calcular Prêmio",
                use_container_width=True,
            )

    # Processa o formulário
    if not submitted:
        st.info("👆 Preencha os dados acima e clique em 'Calcular Prêmio' para obter sua cotação.")
        return

    # Calcula com indicador de progresso
    with st.spinner("🔄 Calculando seu prêmio..."):
        resultado = calcular_premio_atuarial(
            modelo, int(ano), sexo, regiao_desc, faixa_desc
        )

    # Mostra erro se houver
    if resultado["erro"]:
        st.error(f"❌ {resultado['mensagem']}")
        st.info("💡 Tente selecionar outras opções ou entre em contato com nosso suporte.")
        return

    # ============================================================
    # RESULTADO
    # ============================================================
    
    st.success("✅ Cálculo realizado com sucesso!")
    st.markdown("---")
    
    # Seção 1: Prêmio Principal
    st.markdown("### 💰 Valor do Seu Seguro")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.metric(
            label="🎯 Prêmio Estimado",
            value=format_currency(resultado['premio_estimado']),
            help="Valor estimado do seu seguro com base nos dados fornecidos"
        )
    
    with col2:
        st.metric(
            label="📊 Prêmio de Referência",
            value=format_currency(resultado['premio_historico']),
            help="Valor médio praticado no mercado (2019)"
        )
    
    with col3:
        diferenca = resultado['premio_estimado'] - resultado['premio_historico']
        delta_pct = (diferenca / resultado['premio_historico'] * 100) if resultado['premio_historico'] > 0 else 0
        st.metric(
            label="📈 Diferença",
            value=format_currency(abs(diferenca)),
            delta=f"{delta_pct:+.1f}%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Seção 2: Análise de Risco (só mostra se tiver dados)
    risk_label, risk_color = get_risk_level(resultado['frequencia'], resultado['severidade'])
    
    if risk_label is not None:
        st.markdown("### 🎯 Análise de Risco")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Indicadores")
            
            if risk_color == "success":
                st.success(f"**Nível de Risco:** {risk_label}")
            elif risk_color == "warning":
                st.warning(f"**Nível de Risco:** {risk_label}")
            else:
                st.error(f"**Nível de Risco:** {risk_label}")
            
            st.markdown(f"""
            - **Frequência de Sinistros:** {resultado['frequencia']:.4f}
            - **Severidade Média:** {format_currency(resultado['severidade'])}
            """)
        
        with col2:
            st.markdown("#### 🔍 Fatores Considerados")
            st.markdown(f"""
            - ✅ **Modelo:** {resultado['modelo']}
            - ✅ **Ano:** {resultado['ano']}
            - ✅ **Perfil:** {sexo_label}, {resultado['faixa']}
            - ✅ **Região:** {resultado['regiao']}
            """)
        
        st.markdown("---")
    
    # Seção 3: Comparativo de Mercado (SUBSTITUIU EVOLUÇÃO)
    if 'stats_comparativas' in resultado and resultado['stats_comparativas']:
        st.markdown("### 📊 Comparativo de Mercado")
        
        stats = resultado['stats_comparativas']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'premio_medio_modelo' in stats:
                st.metric(
                    "Média do Modelo",
                    format_currency(stats['premio_medio_modelo']),
                    help=f"Média de {stats.get('total_registros', 0)} apólices deste modelo"
                )
        
        with col2:
            if 'premio_min_modelo' in stats:
                st.metric(
                    "Menor Prêmio",
                    format_currency(stats['premio_min_modelo']),
                    help="Menor valor encontrado para este modelo"
                )
        
        with col3:
            if 'premio_max_modelo' in stats:
                st.metric(
                    "Maior Prêmio",
                    format_currency(stats['premio_max_modelo']),
                    help="Maior valor encontrado para este modelo"
                )
        
        st.markdown("---")
    
    # ============================================================
    # EXPLICAÇÃO DA IA - MELHORADA E MAIS COMPLETA
    # ============================================================
    
    st.markdown("### 🤖 Análise Detalhada do Cálculo")
    
    with st.spinner("Gerando análise personalizada..."):
        # Prepara informações para o prompt
        sexo_extenso = "Masculino" if resultado['sexo'] == "M" else "Feminino"
        
        prompt_explicacao = f"""
Você é um consultor especialista em seguros automotivos. Analise o cálculo do prêmio abaixo e explique de forma clara e profissional.

DADOS DO CÁLCULO:
- Modelo do Veículo: {resultado['modelo']}
- Ano de Fabricação: {resultado['ano']}
- Perfil do Condutor: {sexo_extenso}, {resultado['faixa']}
- Região: {resultado['regiao']}
- Prêmio Estimado: R$ {resultado['premio_estimado']:,.2f}
- Prêmio de Referência (Histórico): R$ {resultado['premio_historico']:,.2f}
- Diferença: {((resultado['premio_estimado'] - resultado['premio_historico']) / resultado['premio_historico'] * 100):.1f}%

INSTRUÇÕES PARA SUA RESPOSTA:
1. Escreva em português brasileiro correto, SEM caracteres especiais, acentos mal formatados ou símbolos estranhos
2. NÃO use formatação markdown (sem asteriscos, hashtags, underscores)
3. Escreva em texto corrido, parágrafos normais
4. Estruture sua resposta em 4 parágrafos CLARAMENTE SEPARADOS:

PARÁGRAFO 1 - Visão Geral (3-4 linhas):
INICIE COM: "O prêmio calculado..."
Apresente o valor do prêmio calculado e explique o que ele representa. Mencione que o cálculo considera múltiplos fatores de risco.

PARÁGRAFO 2 - Fatores que Influenciam o Valor (4-5 linhas):
INICIE COM: "Diversos fatores impactaram o valor do prêmio."
Explique DETALHADAMENTE como cada fator impacta o prêmio:
- Idade do veículo (ano {resultado['ano']})
- Perfil do condutor ({sexo_extenso}, faixa etária {resultado['faixa']})
- Localização geográfica ({resultado['regiao']})
- Características do modelo {resultado['modelo']}

PARÁGRAFO 3 - Comparação com Histórico (3-4 linhas):
INICIE COM: "Comparando o prêmio estimado com o histórico..."
Compare o prêmio estimado com o de referência. Explique se o valor está acima, abaixo ou similar ao histórico e possíveis razões para isso.

PARÁGRAFO 4 - Recomendações e Conclusão (3-4 linhas):
INICIE COM: "Como recomendação..."
Forneça orientações práticas sobre o valor calculado. Mencione se é competitivo, quais fatores o cliente pode influenciar para reduzir custos futuros, e que este é um valor estimado baseado em dados históricos.

IMPORTANTE:
- Use APENAS texto simples, sem formatação
- Escreva em português brasileiro perfeito
- Seja profissional mas acessível
- Use números formatados corretamente (R$ 1.234,56)
- NÃO repita os dados que já estão visíveis na tela
- Foque em ANÁLISE e INSIGHTS, não apenas em descrever os números
- INICIE cada parágrafo com a frase indicada acima
"""
        
        try:
            explicacao = llm.invoke(prompt_explicacao)
            texto_explicacao = clean_llm_response(explicacao.content if hasattr(explicacao, 'content') else str(explicacao))
            
            if texto_explicacao and len(texto_explicacao) > 50:
                st.info(texto_explicacao)
            else:
                st.info("""
                O prêmio foi calculado considerando as características do veículo, perfil do condutor e região. 
                O valor estimado está alinhado com as médias históricas do mercado para este perfil específico. 
                Fatores como idade do veículo, experiência do condutor e índices de sinistralidade da região 
                foram considerados no cálculo final.
                """)
        except Exception as e:
            print(f"Erro ao gerar explicação: {e}")
            st.info("""
            O prêmio foi calculado considerando as características do veículo, perfil do condutor e região. 
            O valor estimado está alinhado com as médias históricas do mercado para este perfil específico.
            """)
    
    st.markdown("---")
    
    # Rodapé com disclaimer
    st.markdown("---")
    st.caption("""
    ⚠️ **Aviso Legal:** Este é um valor estimado baseado em dados históricos de 2019. 
    O valor final do seguro pode variar de acordo com análise detalhada da seguradora, 
    coberturas adicionais e condições específicas do veículo.
    """)
    
    # Botão de nova cotação
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Fazer Nova Cotação", use_container_width=True):
            st.rerun()