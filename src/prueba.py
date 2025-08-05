from langchain_google_vertexai import VertexAI

llm = VertexAI(model_name="gemini-pro")

query = "¿Cuál es el impacto de la inteligencia artificial en el sector asegurador?"
response = llm.invoke(query)

print("💬 **Respuesta de VertexAI:**", response)
