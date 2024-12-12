from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


llm = ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=1000,
    verbose=True,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonms for the following word. Return the results as coma separeted list"),
        ("human", "{input}")
    ]
)
chain = prompt | llm


response = chain.invoke({"input": "happy"})
print(response)
