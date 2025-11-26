"""
Módulo para análise de dados auxiliares (acidentes, segurança, demografia)
"""
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from src.app.data_manager import get_data_manager


class AuxiliaryDataAnalyzer:
    """Analisa dados auxiliares para enriquecer contexto"""
    
    def __init__(self):
        self.data_manager = get_data_manager()
        self._mapa_estados = {
            "SP": "São Paulo", "RJ": "Rio de Janeiro", "MG": "Minas Gerais",
            "RS": "Rio Grande do Sul", "PR": "Paraná", "SC": "Santa Catarina",
            "BA": "Bahia", "CE": "Ceará", "PE": "Pernambuco", "AL": "Alagoas",
            "GO": "Goiás", "DF": "Distrito Federal", "ES": "Espírito Santo",
            "PA": "Pará", "AM": "Amazonas", "MA": "Maranhão", "MT": "Mato Grosso",
            "MS": "Mato Grosso do Sul", "RO": "Rondônia", "AC": "Acre",
            "AP": "Amapá", "RR": "Roraima", "TO": "Tocantins", "SE": "Sergipe",
            "PI": "Piauí", "RN": "Rio Grande do Norte", "PB": "Paraíba"
        }
    
    # ============================================================
    # ANÁLISE DE ACIDENTES
    # ============================================================
    
    def get_accident_stats_by_brand(self, marca: str) -> Dict:
        """
        Retorna estatísticas de acidentes para uma marca
        """
        df_acidentes = self.data_manager.get_table("acidentes_2019")
        
        # Normaliza nome da marca
        marca_upper = marca.upper()
        
        # Filtra por marca (busca parcial)
        df_marca = df_acidentes[
            df_acidentes['marca'].str.upper().str.contains(marca_upper, na=False)
        ]
        
        if df_marca.empty:
            return {"encontrado": False, "marca": marca}
        
        # Estatísticas
        total_acidentes = len(df_marca)
        
        # Causas mais comuns
        causas = df_marca['causa_acidente'].value_counts().head(5).to_dict()
        
        # Tipos mais comuns
        tipos = df_marca['tipo_acidente'].value_counts().head(5).to_dict()
        
        # Estados com mais acidentes
        estados = df_marca['uf'].value_counts().head(5).to_dict()
        
        # Perfil dos condutores
        idade_media = df_marca[
            df_marca['tipo_envolvido'] == 'Condutor'
        ]['idade'].mean()
        
        sexo_dist = df_marca[
            df_marca['tipo_envolvido'] == 'Condutor'
        ]['sexo'].value_counts().to_dict()
        
        return {
            "encontrado": True,
            "marca": marca,
            "total_acidentes": total_acidentes,
            "causas_principais": causas,
            "tipos_acidentes": tipos,
            "estados_maior_incidencia": estados,
            "idade_media_condutor": round(idade_media, 1) if pd.notna(idade_media) else None,
            "distribuicao_sexo": sexo_dist
        }
    
    def get_accident_stats_by_state(self, uf: str) -> Dict:
        """
        Retorna estatísticas de acidentes por estado
        """
        df_acidentes = self.data_manager.get_table("acidentes_2019")
        
        uf_upper = uf.upper()
        df_estado = df_acidentes[df_acidentes['uf'] == uf_upper]
        
        if df_estado.empty:
            return {"encontrado": False, "uf": uf}
        
        return {
            "encontrado": True,
            "uf": uf_upper,
            "nome_estado": self._mapa_estados.get(uf_upper, uf_upper),
            "total_acidentes": len(df_estado),
            "causas_principais": df_estado['causa_acidente'].value_counts().head(5).to_dict(),
            "tipos_acidentes": df_estado['tipo_acidente'].value_counts().head(5).to_dict(),
            "marcas_mais_envolvidas": df_estado['marca'].value_counts().head(10).to_dict()
        }
    
    def get_most_dangerous_causes(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna as causas de acidentes mais comuns
        """
        df_acidentes = self.data_manager.get_table("acidentes_2019")
        
        causas = df_acidentes['causa_acidente'].value_counts().head(top_n).reset_index()
        causas.columns = ['causa', 'quantidade']
        
        return causas
    
    def compare_accident_risk(self, marca1: str, marca2: str) -> Dict:
        """
        Compara risco de acidentes entre duas marcas
        """
        stats1 = self.get_accident_stats_by_brand(marca1)
        stats2 = self.get_accident_stats_by_brand(marca2)
        
        if not stats1["encontrado"] or not stats2["encontrado"]:
            return {"erro": "Uma ou ambas as marcas não foram encontradas"}
        
        return {
            "marca1": {
                "nome": marca1,
                "acidentes": stats1["total_acidentes"],
                "causa_principal": list(stats1["causas_principais"].keys())[0] if stats1["causas_principais"] else None
            },
            "marca2": {
                "nome": marca2,
                "acidentes": stats2["total_acidentes"],
                "causa_principal": list(stats2["causas_principais"].keys())[0] if stats2["causas_principais"] else None
            },
            "diferenca_absoluta": abs(stats1["total_acidentes"] - stats2["total_acidentes"]),
            "marca_maior_risco": marca1 if stats1["total_acidentes"] > stats2["total_acidentes"] else marca2
        }
    
    # ============================================================
    # ANÁLISE DE SEGURANÇA PÚBLICA
    # ============================================================
    
    def get_theft_stats_by_state(self, estado: str, ano: Optional[int] = None) -> Dict:
        """
        Retorna estatísticas de roubos/furtos por estado
        """
        df_seg = self.data_manager.get_table("seguranca_publica")
        
        # Normaliza nome do estado
        estado_norm = estado.strip().title()
        
        df_estado = df_seg[df_seg['estado'].str.strip().str.title() == estado_norm]
        
        if df_estado.empty:
            return {"encontrado": False, "estado": estado}
        
        if ano:
            df_estado = df_estado[df_estado['ano'] == str(ano)]
        
        # Separa roubos e furtos
        df_roubo = df_estado[df_estado['tipo_crime'].str.contains('Roubo', na=False)]
        df_furto = df_estado[df_estado['tipo_crime'].str.contains('Furto', na=False)]
        
        return {
            "encontrado": True,
            "estado": estado_norm,
            "ano": ano if ano else "Todos",
            "total_roubos": df_roubo['quantidade'].astype(int).sum(),
            "total_furtos": df_furto['quantidade'].astype(int).sum(),
            "media_mensal_roubos": df_roubo['quantidade'].astype(int).mean(),
            "media_mensal_furtos": df_furto['quantidade'].astype(int).mean(),
            "mes_maior_roubo": df_roubo.loc[df_roubo['quantidade'].astype(int).idxmax(), 'mes'] if not df_roubo.empty else None,
            "mes_maior_furto": df_furto.loc[df_furto['quantidade'].astype(int).idxmax(), 'mes'] if not df_furto.empty else None
        }
    
    def get_most_dangerous_states(self, top_n: int = 10, crime_type: str = "Roubo") -> pd.DataFrame:
        """
        Retorna os estados com mais roubos/furtos
        """
        df_seg = self.data_manager.get_table("seguranca_publica")
        
        # Filtra por tipo de crime
        df_crime = df_seg[df_seg['tipo_crime'].str.contains(crime_type, na=False)]
        
        # Agrupa por estado
        ranking = df_crime.groupby('estado')['quantidade'].sum().nlargest(top_n).reset_index()
        ranking.columns = ['estado', 'total']
        
        return ranking
    
    def get_crime_evolution(self, estado: str) -> pd.DataFrame:
        """
        Retorna evolução temporal de crimes por estado
        """
        df_seg = self.data_manager.get_table("seguranca_publica")
        
        estado_norm = estado.strip().title()
        df_estado = df_seg[df_seg['estado'].str.strip().str.title() == estado_norm]
        
        if df_estado.empty:
            return pd.DataFrame()
        
        # Agrupa por ano e tipo
        evolucao = df_estado.groupby(['ano', 'tipo_crime'])['quantidade'].sum().reset_index()
        
        return evolucao
    
    # ============================================================
    # ANÁLISE DEMOGRÁFICA
    # ============================================================
    
    def get_population_by_state(self, sigla: str, ano: int = 2025) -> Dict:
        """
        Retorna dados populacionais de um estado
        """
        df_pop = self.data_manager.get_table("projecoes_populacao")
        
        sigla_upper = sigla.upper()
        
        # Filtra por UF e ano
        df_estado = df_pop[
            (df_pop['SIGLA'] == sigla_upper) & 
            (df_pop['ANO'] == str(ano))
        ]
        
        if df_estado.empty:
            return {"encontrado": False, "sigla": sigla, "ano": ano}
        
        row = df_estado.iloc[0]
        
        # Calcula proporções de faixas etárias
        pop_total = int(row['POP_T'].replace(',', '')) if isinstance(row['POP_T'], str) else row['POP_T']
        pop_jovem = int(row['15-17_T'].replace(',', '')) if isinstance(row['15-17_T'], str) else row['15-17_T']
        pop_adulta = int(row['18-21_T'].replace(',', '')) if isinstance(row['18-21_T'], str) else row['18-21_T']
        pop_idosa = int(row['60+_T'].replace(',', '')) if isinstance(row['60+_T'], str) else row['60+_T']
        
        return {
            "encontrado": True,
            "local": row['LOCAL'],
            "sigla": sigla_upper,
            "ano": ano,
            "populacao_total": pop_total,
            "populacao_jovem_15_17": pop_jovem,
            "populacao_adulta_18_21": pop_adulta,
            "populacao_idosa_60plus": pop_idosa,
            "proporcao_jovens": (pop_jovem / pop_total * 100) if pop_total > 0 else 0,
            "proporcao_idosos": (pop_idosa / pop_total * 100) if pop_total > 0 else 0
        }
    
    def get_age_distribution_comparison(self, ano: int = 2025) -> pd.DataFrame:
        """
        Compara distribuição etária entre estados
        """
        df_pop = self.data_manager.get_table("projecoes_populacao")
        
        df_ano = df_pop[df_pop['ANO'] == str(ano)]
        
        # Seleciona apenas estados (não Brasil e regiões)
        df_estados = df_ano[df_ano['SIGLA'].str.len() == 2]
        
        if df_estados.empty:
            return pd.DataFrame()
        
        resultado = []
        for _, row in df_estados.iterrows():
            pop_t = int(row['POP_T'].replace(',', '')) if isinstance(row['POP_T'], str) else row['POP_T']
            pop_jovem = int(row['15-17_T'].replace(',', '')) if isinstance(row['15-17_T'], str) else row['15-17_T']
            pop_idosa = int(row['60+_T'].replace(',', '')) if isinstance(row['60+_T'], str) else row['60+_T']
            
            resultado.append({
                'estado': row['LOCAL'],
                'sigla': row['SIGLA'],
                'populacao_total': pop_t,
                'prop_jovens': (pop_jovem / pop_t * 100) if pop_t > 0 else 0,
                'prop_idosos': (pop_idosa / pop_t * 100) if pop_t > 0 else 0
            })
        
        return pd.DataFrame(resultado)
    
    # ============================================================
    # ANÁLISE INTEGRADA
    # ============================================================
    
    def get_integrated_risk_profile(self, modelo: str, uf: str) -> Dict:
        """
        Gera perfil de risco integrado combinando múltiplas fontes
        """
        # Extrai marca do modelo
        marca = modelo.split('/')[0] if '/' in modelo else modelo.split()[0]
        
        # Busca dados de acidentes
        accident_stats = self.get_accident_stats_by_brand(marca)
        accident_state = self.get_accident_stats_by_state(uf)
        
        # Busca dados de segurança
        estado_nome = self._mapa_estados.get(uf.upper(), uf)
        theft_stats = self.get_theft_stats_by_state(estado_nome, 2019)
        
        # Busca dados demográficos
        pop_stats = self.get_population_by_state(uf, 2025)
        
        # Calcula score de risco (0-100)
        risk_score = 50  # Base
        
        if accident_stats.get("encontrado"):
            # Mais acidentes = maior risco
            if accident_stats["total_acidentes"] > 1000:
                risk_score += 15
            elif accident_stats["total_acidentes"] > 500:
                risk_score += 10
            elif accident_stats["total_acidentes"] > 100:
                risk_score += 5
        
        if theft_stats.get("encontrado"):
            # Mais roubos = maior risco
            total_crimes = theft_stats["total_roubos"] + theft_stats["total_furtos"]
            if total_crimes > 10000:
                risk_score += 20
            elif total_crimes > 5000:
                risk_score += 15
            elif total_crimes > 1000:
                risk_score += 10
        
        if pop_stats.get("encontrado"):
            # Mais jovens = potencialmente maior risco
            if pop_stats["proporcao_jovens"] > 5:
                risk_score += 5
        
        risk_score = min(100, risk_score)  # Cap em 100
        
        return {
            "modelo": modelo,
            "uf": uf,
            "estado": estado_nome,
            "risk_score": risk_score,
            "nivel_risco": "Alto" if risk_score > 70 else "Médio" if risk_score > 40 else "Baixo",
            "dados_acidentes": accident_stats,
            "dados_seguranca": theft_stats,
            "dados_demografia": pop_stats,
            "recomendacao": self._generate_recommendation(risk_score)
        }
    
    def _generate_recommendation(self, risk_score: int) -> str:
        """Gera recomendação baseada no score de risco"""
        if risk_score > 70:
            return "Recomendado: Cobertura completa incluindo roubo/furto e assistência 24h"
        elif risk_score > 40:
            return "Recomendado: Cobertura intermediária com proteção contra roubo"
        else:
            return "Recomendado: Cobertura básica pode ser suficiente"


# Instância global
_aux_analyzer = None


def get_auxiliary_analyzer() -> AuxiliaryDataAnalyzer:
    """Retorna instância global do analisador auxiliar"""
    global _aux_analyzer
    if _aux_analyzer is None:
        _aux_analyzer = AuxiliaryDataAnalyzer()
    return _aux_analyzer