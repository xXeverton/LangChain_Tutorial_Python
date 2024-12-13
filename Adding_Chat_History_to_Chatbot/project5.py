from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document 
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

# docA = Document(
#     page_content="The LangChain Expression Language (LCEL) takes a declarative approach to building new Runnables from existing Runnables."
# )

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    # print(len(splitDocs))
    return splitDocs


def create_db(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorStore = FAISS.from_documents(docs,embedding=embeddings)
    return vectorStore


def create_chain(vectoreStore):
    model = ChatGroq(temperature=0.4, model_name="llama-3.1-70b-versatile")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])


    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a seach query to look up in order to get information relevant to the conversation")
    ])

    retriever = vectorStore.as_retriever(search_kwargs={"k":3})
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt,
    )

    retrieval_chain = create_retrieval_chain(
        # retriever, 
        history_aware_retriever,
        chain
    )

    return retrieval_chain


def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        # "context": [docA] 
        "context": docs,
        "chat_history": chat_history
    })

    return response['answer']


if __name__ == '__main__':
    docs = get_documents_from_web("https://python.langchain.com/docs/concepts/lcel/")
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)



