import pdfplumber
import os

docs_path = "data/documents/"

for file in os.listdir(docs_path):
    if file.endswith(".pdf"):
        file_path = os.path.join(docs_path, file)
        print(f"📄 Probando extracción de: {file}")

        try:
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])

            if text.strip():
                print(f"✅ ¡Texto extraído con éxito de {file}!")
                print("🔹 Primeros 500 caracteres extraídos:\n", text[:500])
            else:
                print(f"⚠️ No se pudo extraer texto de {file}. Podría ser un PDF escaneado o corrupto.")

        except Exception as e:
            print(f"❌ Error al extraer texto de {file}: {e}")
