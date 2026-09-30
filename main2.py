from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
llm=ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
history=[SystemMessage(content="you are my tutor helps me in learning everything i need ")]
while True:
    query=input("enter something you want")
    if query.lower()in ["exit","break"]:
     break
    history.append(HumanMessage(content=query))
    response=llm.invoke(history)
    history.append(response)
    print(response.content)
