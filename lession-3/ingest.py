from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


DATA_DIR = Path("data")
INDEX_DIR = "faiss_index"


# 1. Load documents

documents = []

for file_path in DATA_DIR.glob("*.txt"):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    docs = loader.load()

    for doc in docs:
        doc.metadata["category"] = file_path.stem

    documents.extend(docs)


print(f"Loaded {len(documents)} documents")


# 2. Split documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# 3. Create embeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)


# 4. Create vector store

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# 5. Save index

vector_store.save_local(INDEX_DIR)

print(f"Saved FAISS index to '{INDEX_DIR}'")