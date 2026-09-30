from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
llm=ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
response =llm.invoke("who is sachin tendulkar")
print(response.content[0])