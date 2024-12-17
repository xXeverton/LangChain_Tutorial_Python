from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

# Carregar variáveis de ambiente
load_dotenv()

# Configurar o modelo LLM
chat = OllamaLLM(model="llama3:8b")

# Carregar o arquivo CSV
try:
    df = pd.read_csv("fltr3.csv")
except FileNotFoundError:
    print("Erro: O arquivo 'fltr3.csv' não foi encontrado.")
    exit()

parser = StrOutputParser()

# Criar o agente com permissões para executar código perigoso
agent = create_pandas_dataframe_agent(
    chat, 
    df, 
    verbose=True, 
    allow_dangerous_code=True
)

# Consultar o agente
try:
    result = agent.invoke("Conte o tempo total do paciente no hostpital.")
    print("Resposta do agente:", result)
except Exception as e:
    print("Erro ao executar a consulta:", str(e))
