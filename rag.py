from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader

load_dotenv()

# 1. Cargar documento real desde archivo
print("Cargando documento...")
loader = TextLoader("blend_info.txt", encoding="utf-8")
documentos = loader.load()

# 2. Dividir en fragmentos
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
fragmentos = splitter.split_documents(documentos)
print(f"Documento dividido en {len(fragmentos)} fragmentos")

# 3. Crear embeddings y vectorstore
print("Generando embeddings...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(fragmentos, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Prompt
prompt = ChatPromptTemplate.from_template("""
Sos un asistente experto en Blend. Respondé de forma clara y concisa 
basándote solo en el siguiente contexto. Si no sabés algo, decilo.

Contexto:
{context}

Pregunta: {question}

Respuesta:
""")

# 5. Chain RAG
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 6. Preguntas
print("\n=== Sistema RAG con documento real ===\n")
preguntas = [
    "¿Qué casos de uso reales tiene Blend?",
    "¿Qué hace un Senior AI Engineer en Blend?",
    "¿Qué beneficios ofrecen a sus empleados?",
    "¿Con qué tecnologías trabaja Blend?",
]

for pregunta in preguntas:
    respuesta = chain.invoke(pregunta)
    print(f"Q: {pregunta}")
    print(f"A: {respuesta.content}")
    print()
