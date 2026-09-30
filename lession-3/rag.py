from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


INDEX_DIR = "faiss_index"


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)


vector_store = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)


retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


model=ChatGoogleGenerativeAI(
     model="gemini-3.1-flash-lite"
    
    )


def format_documents(documents):

    return "\n\n".join(
        document.page_content
        for document in documents
    )

def retrieve_context(query):

    documents = retriever.invoke(query)

    context = format_documents(documents)

    return context, documents


from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.

Answer the user's question using only the
provided context.

If the answer cannot be found in the context,
say that you don't have enough information.

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])



def answer_question(query):

    context, documents = retrieve_context(query)

    messages = prompt.invoke({
        "context": context,
        "question": query
    })

    response = model.invoke(messages)

    return response, documents




def get_sources(documents):

    return list({
        document.metadata["source"]
        for document in documents
    })

query = "What is the capital of France?"

response, documents = answer_question(query)

print("\nAnswer:")
print(response.content)

print("\nSources:")

for source in get_sources(documents):
    print("-", source)