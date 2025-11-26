from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

# Carrega variáveis do arquivo .env (se existir)
load_dotenv()

# Configurações da sua API Incubator / Azure
api_base = os.getenv("API_BASE")   # 🔹 substitua pelo endpoint correto
api_key = os.getenv("API_KEY")                    # 🔹 substitua pela chave real
azure_deployment = os.getenv("AZURE_DEPLOYMENT")         # nome do deployment criado no Azure
api_version = os.getenv("AZURE_API_VERSION")         # versão recomendada

# Inicializa o modelo Azure OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=api_base,
    api_key=api_key,
    azure_deployment=azure_deployment,
    api_version=api_version,
    temperature=0.7,
    max_tokens=800
)
