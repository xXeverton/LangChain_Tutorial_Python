from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


# llm = ChatGroq(
#     model_name="llama-3.1-70b-versatile",
#     temperature=0.7,
#     max_tokens=1000,
#     verbose=True,
# )

llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")


"""
Diferrentes ways to get response back
    -invoke
    -batch
    -stream (Remember the stream)
"""
response = llm.stream("Hello, how are you")
# print(response)

for chunk in response:
    print(chunk.content, end="", flush=True)