from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.tools.retriever import create_retriever_tool

load_dotenv()

# Create retriever
loader = WebBaseLoader("https://python.langchain.com/docs/concepts/lcel/")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorStore = FAISS.from_documents(docs,embedding=embeddings)
retriever = vectorStore.as_retriever(search_kwargs={"k":3})

model = ChatGroq(temperature=0.7, model_name="llama-3.1-70b-versatile")


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


search = TavilySearchResults()
retriever_tool = create_retriever_tool(
    retriever,
    "lcel_search",
    "Use this tool when searchin for information about Langchain Expression Language (LCEL)"
)
tools = [search, retriever_tool]


agent = create_tool_calling_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)


agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools,
)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history,
    })

    return response["output"]

if __name__ == '__main__':
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)
