from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory


load_dotenv()

UPSTASH_URL="https://infinite-thrush-51489.upstash.io"
UPSTAH_TOKEN="AckhAAIjcDEyMTI3YzZmZmE0ZGM0YjkxYmIyOTNkYThkZTFlMTc3NXAxMA"

history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL,
    token=UPSTAH_TOKEN,
    session_id="chat1",
    ttl=0,
)

model = ChatGroq(temperature=0.7, model_name="llama-3.1-70b-versatile")

prompt = ChatPromptTemplate([
    ("system", "You are frindly AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history,
)

# chain = prompt | model 
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

msg1 = {
    "input": "Meu nome é Everton" 
}
resp1 = chain.invoke(msg1)
print(resp1)


msg2 = {
    "input": "Qual é meu nome?" 
}
resp2 = chain.invoke(msg2)
print(resp2)