from src.driver import driver

rag_chain = driver()

print(rag_chain.invoke({"question": "how much is CGPA"}))