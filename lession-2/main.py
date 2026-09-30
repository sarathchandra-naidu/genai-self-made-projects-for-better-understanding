from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader,TextLoader
loader=DirectoryLoader("data",glob="**/*.txt",loader_cls=TextLoader)
docs=loader.load()
# print(docs[0].page_content)

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)
chunks=splitter.split_documents(docs)
print(chunks)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
# vectors=embeddings.embed_documents(chunks)

from langchain_community.vectorstores import FAISS
vector_store=FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
query=input("enter something you want")
response=vector_store.similarity_search(
    query,
    k=3
)
print(response)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
result=retriever.invoke(query)

# persistance whenever you run a program we always building everything from scratch but we can save the done work
vector_store.save_local("faiss_index")

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)