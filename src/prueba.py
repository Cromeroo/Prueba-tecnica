from langchain_google_vertexai import VertexAI

# 🔹 Cargar el modelo directamente
llm = VertexAI(model_name="gemini-pro")

# 🔹 Prueba con una consulta de prueba
query = "¿Cuál es el impacto de la inteligencia artificial en el sector asegurador?"
response = llm.invoke(query)

print("💬 **Respuesta de VertexAI:**", response)
