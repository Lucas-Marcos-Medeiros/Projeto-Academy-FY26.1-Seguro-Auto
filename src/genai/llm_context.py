"""
Módulo para enriquecer o contexto da LLM com dados de múltiplas tabelas
"""
import pandas as pd
from typing import Dict, List, Optional
from src.app.data_manager import get_data_manager
from src.analises.auxiliary_data_analyzer import get_auxiliary_analyzer


class LLMContextEnricher:
    """Enriquece prompts da LLM com contexto de múltiplas tabelas"""
    
    def __init__(self):
        self.data_manager = get_data_manager()
        self.aux_analyzer = get_auxiliary_analyzer()
        
    def extract_intent(self, user_message: str) -> Dict:
        """
        Extrai a intenção do usuário e entidades mencionadas
        (versão simples - pode ser melhorada com NLP)
        """
        message_lower = user_message.lower()
        
        intent = {
            "needs_data": False,
            "tables_needed": [],
            "entities": {},
            "query_type": "general"
        }
        
        # Detecta menções a modelos de carros
        modelos = self.data_manager.get_unique_values("casco", "modelo")
        for modelo in modelos:
            if modelo.lower() in message_lower:
                intent["entities"]["modelo"] = modelo
                intent["needs_data"] = True
                intent["tables_needed"].append("casco")
                
        # Detecta menções a regiões
        if any(word in message_lower for word in ["região", "regiao", "sp", "rj", "mg"]):
            intent["needs_data"] = True
            intent["tables_needed"].append("casco")
            
        # Detecta perguntas sobre prêmios
        if any(word in message_lower for word in ["prêmio", "premio", "preço", "preco", "valor", "custo"]):
            intent["query_type"] = "pricing"
            intent["needs_data"] = True
            intent["tables_needed"].append("casco")
            
        # Detecta perguntas sobre sinistros
        if any(word in message_lower for word in ["sinistro", "acidente", "colisão", "roubo"]):
            intent["query_type"] = "claims"
            intent["needs_data"] = True
            
        return intent
    
    def get_relevant_data(self, intent: Dict) -> Dict[str, pd.DataFrame]:
        """Busca dados relevantes baseado na intenção"""
        data = {}
        
        if not intent["needs_data"]:
            return data
            
        # Busca dados combinados dos dois semestres
        if "casco" in intent["tables_needed"]:
            df_combined = self.data_manager.get_combined_casco_data()
            
            # Filtra por modelo se mencionado
            if "modelo" in intent["entities"]:
                modelo = intent["entities"]["modelo"]
                df_filtered = df_combined[df_combined["modelo"] == modelo]
                data["casco_filtered"] = df_filtered
                
                # Separa por semestre para análise temporal
                data["casco_sem1"] = df_filtered[df_filtered['semestre'] == 1]
                data["casco_sem2"] = df_filtered[df_filtered['semestre'] == 2]
            else:
                # Retorna amostra se não houver filtro específico
                data["casco_sample"] = df_combined.sample(min(10, len(df_combined)))
        
        # Busca dados de acidentes se relevante
        if "acidentes" in intent["tables_needed"]:
            if "modelo" in intent["entities"]:
                modelo = intent["entities"]["modelo"]
                # Extrai marca do modelo
                marca = modelo.split('/')[0] if '/' in modelo else modelo.split()[0]
                
                accident_stats = self.aux_analyzer.get_accident_stats_by_brand(marca)
                if accident_stats.get("encontrado"):
                    data["acidentes_info"] = accident_stats
        
        # Busca dados de segurança se relevante
        if "seguranca" in intent["tables_needed"]:
            if "estado_mencionado" in intent["entities"]:
                keyword = intent["entities"]["estado_mencionado"]
                
                # Mapeia keyword para nome de estado
                estado_map = {
                    "sp": "São Paulo", "são paulo": "São Paulo",
                    "rj": "Rio de Janeiro", "rio de janeiro": "Rio de Janeiro",
                    "mg": "Minas Gerais", "minas": "Minas Gerais"
                }
                
                estado = estado_map.get(keyword, keyword.title())
                theft_stats = self.aux_analyzer.get_theft_stats_by_state(estado, 2019)
                
                if theft_stats.get("encontrado"):
                    data["seguranca_info"] = theft_stats
                
        return data
    
    def format_data_for_llm(self, data: Dict) -> str:
        """Formata os dados para incluir no prompt da LLM"""
        if not data:
            return ""
            
        context = "\n\n📊 **DADOS RELEVANTES:**\n\n"
        
        # Formata dados de DataFrames
        for table_name, content in data.items():
            if isinstance(content, pd.DataFrame):
                df = content
                if df.empty:
                    continue
                
                # Identifica se é dado semestral
                if "sem1" in table_name:
                    context += f"**{table_name}** - 1º Semestre 2019 ({len(df)} registros):\n"
                elif "sem2" in table_name:
                    context += f"**{table_name}** - 2º Semestre 2019 ({len(df)} registros):\n"
                else:
                    context += f"**{table_name}** ({len(df)} registros):\n"
                
                # Colunas mais relevantes
                relevant_cols = [
                    "modelo", "ano", "sexo", "regiao_desc", "faixa_desc",
                    "premio1", "freq_sin1", "indeniz1", "periodo"
                ]
                
                display_cols = [col for col in relevant_cols if col in df.columns]
                
                if len(df) <= 5:
                    context += df[display_cols].to_string(index=False)
                else:
                    context += "Estatísticas:\n"
                    numeric_cols = df[display_cols].select_dtypes(include='number').columns
                    
                    for col in numeric_cols:
                        context += f"  - {col}: média={df[col].mean():.2f}, "
                        context += f"min={df[col].min():.2f}, max={df[col].max():.2f}\n"
                        
                context += "\n"
        
        # Formata dados de acidentes
        if "acidentes_info" in data:
            acc = data["acidentes_info"]
            context += "🚗 **DADOS DE ACIDENTES:**\n"
            context += f"  - Total de acidentes (2019): {acc['total_acidentes']}\n"
            
            if acc.get("causas_principais"):
                context += "  - Causas principais:\n"
                for causa, qtd in list(acc["causas_principais"].items())[:3]:
                    context += f"    * {causa}: {qtd} casos\n"
            
            if acc.get("tipos_acidentes"):
                context += "  - Tipos mais comuns:\n"
                for tipo, qtd in list(acc["tipos_acidentes"].items())[:3]:
                    context += f"    * {tipo}: {qtd} casos\n"
            
            context += "\n"
        
        # Formata dados de segurança
        if "seguranca_info" in data:
            seg = data["seguranca_info"]
            context += "🔒 **DADOS DE SEGURANÇA:**\n"
            context += f"  - Estado: {seg.get('estado', 'N/A')}\n"
            context += f"  - Total de roubos (2019): {seg.get('total_roubos', 0):,}\n"
            context += f"  - Total de furtos (2019): {seg.get('total_furtos', 0):,}\n"
            context += f"  - Média mensal de roubos: {seg.get('media_mensal_roubos', 0):.0f}\n"
            context += "\n"
        
        # Adiciona análise comparativa se houver dados de ambos semestres
        if "casco_sem1" in data and "casco_sem2" in data:
            df1 = data["casco_sem1"]
            df2 = data["casco_sem2"]
            
            if not df1.empty and not df2.empty and "premio1" in df1.columns:
                media_sem1 = df1["premio1"].mean()
                media_sem2 = df2["premio1"].mean()
                variacao = ((media_sem2 - media_sem1) / media_sem1 * 100) if media_sem1 > 0 else 0
                
                context += "📈 **ANÁLISE TEMPORAL:**\n"
                context += f"- Prêmio médio 1º Sem: R$ {media_sem1:.2f}\n"
                context += f"- Prêmio médio 2º Sem: R$ {media_sem2:.2f}\n"
                context += f"- Variação: {variacao:+.2f}%\n\n"
            
        return context
    
    def enrich_prompt(self, user_message: str, chat_history: List = None) -> str:
        """
        Enriquece o prompt do usuário com contexto relevante
        """
        # Extrai intenção
        intent = self.extract_intent(user_message)
        
        # Busca dados relevantes
        relevant_data = self.get_relevant_data(intent)
        
        # Formata dados
        data_context = self.format_data_for_llm(relevant_data)
        
        # Monta prompt enriquecido
        enriched_prompt = f"""Você é um assistente especializado em seguros automotivos.

{data_context}

**Instruções:**
- Use os dados fornecidos acima para responder de forma precisa
- Se não houver dados suficientes, seja honesto sobre as limitações
- Forneça respostas claras e objetivas
- Quando relevante, mencione tendências nos dados

**Histórico da conversa:**
{self._format_history(chat_history) if chat_history else "Nenhum histórico anterior."}

**Pergunta do usuário:**
{user_message}

**Sua resposta:**"""
        
        return enriched_prompt
    
    def _format_history(self, history: List) -> str:
        """Formata o histórico de mensagens"""
        if not history:
            return ""
            
        formatted = []
        for role, content in history[-5:]:  # Últimas 5 mensagens
            formatted.append(f"{role.upper()}: {content}")
            
        return "\n".join(formatted)
    
    def get_calculator_context(self, 
                               modelo: str, 
                               ano: int, 
                               sexo: str,
                               regiao_desc: str, 
                               faixa_desc: str) -> Dict:
        """
        Busca contexto adicional para a calculadora
        """
        context = {
            "tabelas_usadas": ["casco_sem1", "casco_sem2"],
            "dados_complementares": {}
        }
        
        # Dados combinados
        df_combined = self.data_manager.get_combined_casco_data()
        
        # Estatísticas do modelo (combinando ambos semestres)
        df_modelo = df_combined[df_combined["modelo"] == modelo]
        if not df_modelo.empty:
            context["dados_complementares"]["estatisticas_modelo"] = {
                "total_registros": len(df_modelo),
                "premio_medio": df_modelo["premio1"].mean(),
                "premio_min": df_modelo["premio1"].min(),
                "premio_max": df_modelo["premio1"].max(),
            }
            
            # Estatísticas por semestre
            df_sem1 = df_modelo[df_modelo['semestre'] == 1]
            df_sem2 = df_modelo[df_modelo['semestre'] == 2]
            
            if not df_sem1.empty and not df_sem2.empty:
                context["dados_complementares"]["evolucao_temporal"] = {
                    "premio_medio_sem1": df_sem1["premio1"].mean(),
                    "premio_medio_sem2": df_sem2["premio1"].mean(),
                    "variacao": ((df_sem2["premio1"].mean() - df_sem1["premio1"].mean()) / 
                                df_sem1["premio1"].mean() * 100) if df_sem1["premio1"].mean() > 0 else 0
                }
        
        # Estatísticas da região
        df_regiao = df_combined[df_combined["regiao_desc"] == regiao_desc]
        if not df_regiao.empty:
            context["dados_complementares"]["estatisticas_regiao"] = {
                "total_registros": len(df_regiao),
                "premio_medio": df_regiao["premio1"].mean(),
                "modelos_populares": df_regiao["modelo"].value_counts().head(3).to_dict()
            }
            
        return context


# Instância global
_enricher = None


def get_context_enricher() -> LLMContextEnricher:
    """Retorna a instância global do enriquecedor"""
    global _enricher
    if _enricher is None:
        _enricher = LLMContextEnricher()
    return _enricher