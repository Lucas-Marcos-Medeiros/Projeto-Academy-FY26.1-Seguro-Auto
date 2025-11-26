"""
Módulo para carregar datasets do HuggingFace
VERSÃO CORRIGIDA - Com tratamento para notação científica em acidentes_2019
"""
import pandas as pd
import os
import streamlit as st
from typing import Optional, Dict
import io


class HuggingFaceLoader:
    """Carrega datasets do HuggingFace Hub"""
    
    def __init__(self):
        self.dataset_repo = self._get_dataset_repo()
        self.file_configs = self._get_file_configs()
        
    def _get_dataset_repo(self) -> str:
        """Obtém o nome do repositório do dataset"""
        repo = os.getenv('HF_DATASET_REPO')
        if repo:
            return repo

        # Fallback padrão
        return "Qluks/seguros-dataset"
    
    def _get_file_configs(self) -> Dict:
        """Configuração dos arquivos no dataset"""
        return {
            'casco_sem1': {
                'filename': 'casco_tratadoA.parquet',
                'separator': ',',
                'encoding': 'utf-8',
                'description': '1º Semestre 2019'
            },
            'casco_sem2': {
                'filename': 'casco_tratadoB.parquet',
                'separator': ',',
                'encoding': 'utf-8',
                'description': '2º Semestre 2019'
            },
            'acidentes_2019': {
                'filename': 'acidentes2019_todas_causas_tipos.parquet',
                'separator': ';',
                'encoding': 'latin1',
                'description': 'Acidentes 2019',
                'has_scientific_notation': True  # ← NOVA FLAG
            },
            'seguranca_publica': {
                'filename': 'indicadoressegurancapublicauf.parquet',
                'separator': ',',
                'encoding': 'utf-8',
                'description': 'Segurança Pública',
                'has_header': False,
                'column_names': ['estado', 'tipo_crime', 'ano', 'mes', 'quantidade']
            },
            'projecoes_populacao': {
                'filename': 'projecoes_grupos_etarios_quantidades.parquet',
                'separator': ',',
                'encoding': 'utf-8',
                'description': 'Projeções Populacionais'
            }
        }
    
    @st.cache_data(ttl=3600)
    def load_parquet(_self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Carrega Parquet do HuggingFace de forma correta
        """
        if table_name not in _self.file_configs:
            st.error(f"Tabela não configurada: {table_name}")
            return None
        
        config = _self.file_configs[table_name]
        filename = config['filename']

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                _self.dataset_repo,
                data_files=filename,
                split="train"
            )

            df = dataset.to_pandas()

            # limpeza (válida para parquet)
            df = _self._clean_dataframe(df, table_name)

            print(f"✅ {filename}: {len(df)} registros, {len(df.columns)} colunas")
            return df

        except Exception as e:
            st.error(f"Erro ao carregar {filename}: {e}")
            print("❌ Erro detalhado:", e)
            return None

    
    def _clean_dataframe(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Limpa e formata o DataFrame"""
        
        # Remove colunas completamente vazias
        df = df.dropna(axis=1, how='all')
        
        # Remove linhas completamente vazias
        df = df.dropna(axis=0, how='all')
        
        # Limpa nomes de colunas (remove espaços, caracteres especiais)
        df.columns = df.columns.str.strip()
        
        # Correções específicas por tabela
        if table_name == 'acidentes_2019':
            # Garante que não há colunas duplicadas
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Carrega todas as tabelas configuradas"""
        tables = {}
        
        total = len(self.file_configs)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (table_name, config) in enumerate(self.file_configs.items()):
            status_text.text(f"Carregando {config['description']}... ({i+1}/{total})")
            
            df = self.load_parquet(table_name)
            
            if df is not None:
                tables[table_name] = df
                print(f"✅ {table_name}: {len(df):,} registros")
            else:
                print(f"⚠️ {table_name}: Falha ao carregar")
            
            progress_bar.progress((i + 1) / total)
        
        status_text.text("✅ Carregamento concluído!")
        progress_bar.empty()
        status_text.empty()
        
        return tables
    
    def get_info(self) -> Dict:
        """Retorna informações sobre o dataset"""
        return {
            'dataset_repo': self.dataset_repo,
            'configured': bool(self.dataset_repo),
            'files': {
                name: config['description'] 
                for name, config in self.file_configs.items()
            }
        }


# Instância global
_hf_loader = None


def get_huggingface_loader() -> HuggingFaceLoader:
    """Retorna instância global do loader"""
    global _hf_loader
    if _hf_loader is None:
        _hf_loader = HuggingFaceLoader()
    return _hf_loader


def check_huggingface_config() -> bool:
    """Verifica se HuggingFace está configurado"""
    loader = get_huggingface_loader()
    info = loader.get_info()
    
    if not info['configured']:
        st.warning("⚠️ HuggingFace não está configurado")
        with st.expander("Como configurar"):
            st.markdown("""
            **1. Configure o repositório do dataset:**
            
            Em `.streamlit/secrets.toml`:
            ```toml
            [huggingface]
            dataset_repo = "Pichau2907/casco_dataset"
            ```
            
            Ou como variável de ambiente:
            ```bash
            export HF_DATASET_REPO="Pichau2907/casco_dataset"
            ```
            
            **2. Instale a biblioteca:**
            ```bash
            pip install datasets
            ```
            """)
        return False
    
    st.success(f"✅ Dataset: {info['dataset_repo']}")
    return True


# Script de teste
if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTE DO HUGGINGFACE LOADER")
    print("="*60)
    
    loader = get_huggingface_loader()
    info = loader.get_info()
    
    print(f"\nDataset: {info['dataset_repo']}")
    print(f"Configurado: {info['configured']}")
    
    print("\n📋 Tabelas disponíveis:")
    for name, desc in info['files'].items():
        print(f"  • {name}: {desc}")
    
    # Testa carregamento de uma tabela problemática
    print("\n" + "="*60)
    print("🧪 TESTANDO ACIDENTES_2019 (com notação científica)")
    print("="*60)
    
    try:
        df = loader.load_parquet('acidentes_2019')
        if df is not None:
            print(f"\n✅ Carregado com sucesso!")
            print(f"   Registros: {len(df)}")
            print(f"   Colunas: {df.columns.tolist()[:10]}...")
            print(f"\n📊 Primeiras linhas:")
            print(df.head())
            print(f"\n📊 Tipos de dados:")
            print(df.dtypes)
        else:
            print("\n❌ Falha ao carregar")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()