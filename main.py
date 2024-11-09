from src.driver import driver

driver_instance = driver()

rag_chain = driver_instance.get_rag_chain()

print(rag_chain.invoke({"question": "how much is CGPA"}))

