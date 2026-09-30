from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import List
class moviesummary(BaseModel):
    totalmovies:list[str]
    hitmovies:list[str]
    flopmovies:list[str]
    name:str
    
llm=ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
model=llm.with_structured_output(moviesummary)
response =model.invoke("mahesh babu")
print(response)