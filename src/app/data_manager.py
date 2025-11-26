"""
Gerenciador centralizado para múltiplas tabelas de dados
Versão CORRIGIDA - Usa APENAS HuggingFace (sem arquivos locais)
"""
import pandas as pd
import streamlit as st
import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TableConfig:
    """Configuração de uma tabela"""
    name: str
    file_path: str
    description: str
    key_columns: List[str]


class DataManager:
    """Gerencia o carregamento e acesso a múltiplas tabelas"""
    
    def __init__(self, use_huggingface: bool = True):
        """
        Inicializa o DataManager
        
        Args:
            use_huggingface: Sempre True (não usa mais arquivos locais)
        """
        self.use_huggingface = use_huggingface
        self.tables: Dict[str, pd.DataFrame] = {}
        self.configs: Dict[str, TableConfig] = {}
        
        # Carrega loader do HuggingFace
        if use_huggingface:
            try:
                from src.utils.huggingface_loader import get_huggingface_loader
                self.hf_loader = get_huggingface_loader()
                print("✅ HuggingFace loader habilitado")
            except ImportError:
                print("⚠️ Tentando import alternativo...")
                try:
                    from src.utils.huggingface_loader import get_huggingface_loader
                    self.hf_loader = get_huggingface_loader()
                    print("✅ HuggingFace loader habilitado (src.utils)")
                except ImportError:
                    print("❌ Módulo huggingface_loader não encontrado")
                    print("💡 Certifique-se de que huggingface_loader.py está no mesmo diretório")
                    self.use_huggingface = False
                    self.hf_loader = None
        else:
            self.hf_loader = None
        
    def register_table(self, config: TableConfig):
        """Registra uma nova tabela no sistema"""
        self.configs[config.name] = config
        
    def load_table(self, table_name: str) -> pd.DataFrame:
        if table_name in self.tables:
            return self.tables[table_name]

        if table_name not in self.configs:
            raise ValueError(f"Tabela '{table_name}' não registrada")

        @st.cache_data(show_spinner=False)
        def load_cached_parquet(table_name):
            print(f"📥 Baixando {table_name} via HuggingFace (primeira vez)...")
            return self.hf_loader.load_parquet(table_name)

        df = load_cached_parquet(table_name)
        self.tables[table_name] = df
        return df
    
    def _load_from_huggingface(self, table_name: str) -> Optional[pd.DataFrame]:
        """Carrega tabela do HuggingFace"""
        try:
            print(f"🔄 Carregando {table_name} do HuggingFace...")
            
            # Usa o loader do HuggingFace
            df = self.hf_loader.load_parquet(table_name)
            
            if df is not None:
                print(f"✅ {table_name}: {len(df)} registros, {len(df.columns)} colunas")
                return df
            else:
                print(f"❌ Falha ao carregar {table_name}")
                return None
            
        except Exception as e:
            print(f"❌ Erro ao carregar {table_name} do HuggingFace: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_table(self, table_name: str) -> pd.DataFrame:
        """Retorna uma tabela carregada"""
        if table_name not in self.tables:
            self.load_table(table_name)
        return self.tables[table_name]
    
    def load_all_tables(self):
        """Carrega todas as tabelas registradas"""
        print("\n" + "="*60)
        print("📊 CARREGANDO TODAS AS TABELAS DO HUGGINGFACE")
        print("="*60 + "\n")
        
        for table_name in self.configs.keys():
            try:
                self.load_table(table_name)
            except Exception as e:
                print(f"⚠️ Falha ao carregar {table_name}: {e}")
        
        print("\n" + "="*60)
        print(f"✅ Carregamento concluído: {len(self.tables)}/{len(self.configs)} tabelas")
        print("="*60 + "\n")
    
    def get_table_info(self, table_name: str) -> Dict:
        """Retorna informações sobre uma tabela"""
        df = self.get_table(table_name)
        config = self.configs[table_name]
        
        return {
            "name": config.name,
            "description": config.description,
            "rows": len(df),
            "columns": list(df.columns),
            "key_columns": config.key_columns,
            "sample": df.head(3).to_dict('records')
        }
    
    def get_all_tables_summary(self) -> str:
        """Retorna um resumo de todas as tabelas disponíveis"""
        summary = "📊 **Tabelas Disponíveis:**\n\n"
        
        for table_name, config in self.configs.items():
            try:
                df = self.get_table(table_name)
                summary += f"**{config.name}**\n"
                summary += f"- Descrição: {config.description}\n"
                summary += f"- Registros: {len(df):,}\n"
                summary += f"- Colunas principais: {', '.join(config.key_columns[:3])}\n\n"
            except Exception as e:
                summary += f"**{config.name}**\n"
                summary += f"- ⚠️ Erro ao carregar: {str(e)}\n\n"
            
        return summary
    
    def query_tables(self, filters: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Filtra múltiplas tabelas com critérios específicos
        
        Args:
            filters: Dict no formato {table_name: {column: value}}
        
        Returns:
            Dict com tabelas filtradas
        """
        results = {}
        
        for table_name, conditions in filters.items():
            df = self.get_table(table_name)
            
            for column, value in conditions.items():
                if column in df.columns:
                    df = df[df[column] == value]
            
            results[table_name] = df
        
        return results
    
    def combine_tables(self, table_names: List[str], add_period_column: bool = True) -> pd.DataFrame:
        """Combina múltiplas tabelas verticalmente"""
        dfs = []
        
        for i, table_name in enumerate(table_names):
            df = self.get_table(table_name).copy()
            
            if add_period_column:
                if "sem1" in table_name:
                    df['periodo'] = "1º Semestre 2019"
                    df['semestre'] = 1
                elif "sem2" in table_name:
                    df['periodo'] = "2º Semestre 2019"
                    df['semestre'] = 2
                else:
                    df['periodo'] = table_name
                    df['semestre'] = i + 1
                    
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"✅ Tabelas combinadas: {len(combined)} registros totais")
        return combined
    
    def get_combined_casco_data(self) -> pd.DataFrame:
        """
        Retorna dados combinados dos dois semestres
        VERSÃO CORRIGIDA - Não tenta carregar arquivos locais
        """
        cache_key = "casco_combined"
        
        if cache_key not in self.tables:
            try:
                print("\n🔄 Carregando dados combinados de casco...")
                
                # Carrega ambas as tabelas do HuggingFace
                df_sem1 = self.get_table("casco_sem1")
                df_sem2 = self.get_table("casco_sem2")
                
                print(f"✅ Semestre 1: {len(df_sem1)} registros")
                print(f"✅ Semestre 2: {len(df_sem2)} registros")
                
                # Adiciona coluna de período
                df_sem1 = df_sem1.copy()
                df_sem1['periodo'] = "1º Semestre 2019"
                df_sem1['semestre'] = 1
                
                df_sem2 = df_sem2.copy()
                df_sem2['periodo'] = "2º Semestre 2019"
                df_sem2['semestre'] = 2
                
                # Combina
                self.tables[cache_key] = pd.concat([df_sem1, df_sem2], ignore_index=True)
                print(f"✅ Dados combinados: {len(self.tables[cache_key])} registros\n")
                
            except Exception as e:
                print(f"❌ Erro ao combinar tabelas: {e}")
                import traceback
                traceback.print_exc()
                
                # Tenta apenas o primeiro semestre como fallback
                try:
                    print("⚠️ Tentando carregar apenas 1º semestre...")
                    df = self.get_table("casco_sem1").copy()
                    df['periodo'] = "1º Semestre 2019"
                    df['semestre'] = 1
                    self.tables[cache_key] = df
                    print(f"⚠️ Usando apenas 1º semestre: {len(df)} registros")
                except Exception as e2:
                    print(f"❌ Falha ao carregar 1º semestre: {e2}")
                    return pd.DataFrame()
            
        return self.tables.get(cache_key, pd.DataFrame())
    
    def compare_periods(self, metric: str, group_by: List[str], 
                       table1: str = "casco_sem1", table2: str = "casco_sem2") -> pd.DataFrame:
        """Compara uma métrica entre dois períodos"""
        df1 = self.get_table(table1)
        df2 = self.get_table(table2)
        
        agg1 = df1.groupby(group_by)[metric].mean().reset_index()
        agg1.columns = list(group_by) + [f'{metric}_sem1']
        
        agg2 = df2.groupby(group_by)[metric].mean().reset_index()
        agg2.columns = list(group_by) + [f'{metric}_sem2']
        
        comparison = pd.merge(agg1, agg2, on=group_by, how='outer')
        comparison['variacao_absoluta'] = comparison[f'{metric}_sem2'] - comparison[f'{metric}_sem1']
        comparison['variacao_percentual'] = (
            (comparison['variacao_absoluta'] / comparison[f'{metric}_sem1']) * 100
        ).round(2)
        
        return comparison
    
    def get_unique_values(self, table_name: str, column: str) -> List:
        """Retorna valores únicos de uma coluna"""
        if table_name == "casco":
            df = self.get_combined_casco_data()
        else:
            df = self.get_table(table_name)
            
        if column not in df.columns:
            return []
        return sorted(df[column].dropna().unique().tolist())


# Instância global
_data_manager = None


def get_data_manager(use_huggingface: bool = True) -> DataManager:
    """
    Retorna a instância global do DataManager
    VERSÃO CORRIGIDA - Não usa mais arquivos locais
    """
    global _data_manager
    
    if _data_manager is None:
        print("🚀 Inicializando DataManager com HuggingFace...")
        _data_manager = DataManager(use_huggingface=use_huggingface)
        _initialize_tables(_data_manager)
        
    return _data_manager


def _initialize_tables(manager: DataManager):
    """Inicializa as configurações das tabelas"""
    
    print("📝 Registrando tabelas...")
    
    # Tabelas principais
    manager.register_table(TableConfig(
        name="casco_sem1",
        file_path="casco_tratadoA.parquet",
        description="Dados de seguros de casco - 1º Semestre 2019",
        key_columns=["modelo", "ano", "sexo", "regiao_desc", "faixa_desc"]
    ))
    
    manager.register_table(TableConfig(
        name="casco_sem2",
        file_path="casco_tratadoB.parquet",
        description="Dados de seguros de casco - 2º Semestre 2019",
        key_columns=["modelo", "ano", "sexo", "regiao_desc", "faixa_desc"]
    ))
    
    # Tabelas auxiliares
    manager.register_table(TableConfig(
        name="acidentes_2019",
        file_path="acidentes2019_todas_causas_tipos.parquet",
        description="Dados de acidentes de trânsito em 2019",
        key_columns=["uf", "causa_acidente", "tipo_acidente", "marca"]
    ))
    
    manager.register_table(TableConfig(
        name="seguranca_publica",
        file_path="indicadoressegurancapublicauf.parquet",
        description="Indicadores de roubos e furtos por estado",
        key_columns=["estado", "tipo_crime", "ano", "mes"]
    ))
    
    manager.register_table(TableConfig(
        name="projecoes_populacao",
        file_path="projecoes_grupos_etarios_quantidades.parquet",
        description="Projeções demográficas por faixa etária",
        key_columns=["ANO", "SIGLA", "LOCAL"]
    ))
    
    print(f"✅ {len(manager.configs)} tabelas registradas\n")


# Script de teste
if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTE DO DATA MANAGER")
    print("="*60 + "\n")
    
    # Inicializa
    dm = get_data_manager()
    
    # Mostra resumo
    print("\n" + dm.get_all_tables_summary())
    
    # Testa dados combinados
    print("\n" + "="*60)
    print("🔄 TESTANDO DADOS COMBINADOS")
    print("="*60 + "\n")
    
    combined = dm.get_combined_casco_data()
    if not combined.empty:
        print(f"✅ Dados combinados: {len(combined)} registros")
        print(f"📊 Colunas: {combined.columns.tolist()}")
        print(f"\n📋 Primeiras linhas:")
        print(combined.head())
    else:
        print("❌ Nenhum dado combinado")