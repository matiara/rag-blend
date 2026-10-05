from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import datetime

load_dotenv()

# ── HERRAMIENTAS ──────────────────────────────────────────────────────────────

@tool
def calcular(expresion: str) -> str:
    """Calcula una expresión matemática. Ejemplo: 2 + 2 o 150 * 0.21"""
    try:
        return str(eval(expresion))
    except:
        return "No pude calcular esa expresión"

@tool
def fecha_actual(input: str = "") -> str:
    """Devuelve la fecha y hora actual"""
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

@tool
def buscar_info(query: str) -> str:
    """Busca información sobre un tema. Usala cuando necesites datos que no tenés."""
    respuestas = {
        "langchain": "LangChain es un framework open source para construir aplicaciones con LLMs. Permite conectar modelos de lenguaje con herramientas, bases de datos y APIs.",
        "blend": "Blend es una empresa de servicios de IA que construye soluciones con RAG, agentes y LLMs para clientes Fortune 500.",
        "rag": "RAG (Retrieval Augmented Generation) es una arquitectura que combina búsqueda semántica con generación de texto para responder preguntas con información real.",
    }
    for key, val in respuestas.items():
        if key in query.lower():
            return val
    return f"Información sobre '{query}': Es un tema relevante en el campo de AI Engineering."

tools = [calcular, fecha_actual, buscar_info]

# ── AGENTE ────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agente = create_react_agent(llm, tools)

# ── PRUEBAS ───────────────────────────────────────────────────────────────────
print("=== AGENTE CON HERRAMIENTAS ===\n")

preguntas = [
    "¿Qué día y hora es ahora?",
    "¿Cuánto es el 23% de 1500?",
    "¿Qué es LangChain?",
    "¿Qué es RAG?",
]

for pregunta in preguntas:
    print(f"Pregunta: {pregunta}")
    print("-" * 40)
    resultado = agente.invoke({"messages": [HumanMessage(content=pregunta)]})
    ultima_respuesta = resultado["messages"][-1].content
    print(f"Respuesta: {ultima_respuesta}\n")
