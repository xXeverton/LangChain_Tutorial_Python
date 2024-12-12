"""
Retrieval chain 
"""

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


load_dotenv()

# docA = Document(
#     page_content="The LangChain Expression Language (LCEL) takes a declarative approach to building new Runnables from existing Runnables."
# )

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
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

    prompt = ChatPromptTemplate.from_template("""
    Awser the user's question:
    Context: {context}
    Question: {input}
    """)


    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k":1})
    retrieval_chain = create_retrieval_chain(
        retriever, 
        chain
    )

    return retrieval_chain


docs = get_documents_from_web("https://python.langchain.com/docs/concepts/lcel/")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)



response = chain.invoke({
    "input": "What is LCEL?",
    # "context": [docA] 
    "context": docs
})

print(response['answer'])