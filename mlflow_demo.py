from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import mlflow
import time

load_dotenv()

# Iniciamos MLflow
mlflow.set_experiment("blend-prompt-experiments")

# Los prompts que vamos a comparar
prompts = [
    {
        "nombre": "prompt_basico",
        "template": "Respondé esta pregunta: {pregunta}"
    },
    {
        "nombre": "prompt_experto", 
        "template": """Sos un experto en AI Engineering con 10 años de experiencia.
Respondé de forma clara, técnica y concisa.
Pregunta: {pregunta}"""
    },
    {
        "nombre": "prompt_estructurado",
        "template": """Respondé la siguiente pregunta en este formato exacto:
RESPUESTA CORTA: (máximo 1 oración)
DETALLE: (2-3 oraciones de profundidad)
EJEMPLO: (un ejemplo concreto)

Pregunta: {pregunta}"""
    },
]

pregunta = "¿Qué es un sistema RAG y para qué sirve?"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("=== EXPERIMENTOS CON MLFLOW ===\n")

for prompt_config in prompts:
    with mlflow.start_run(run_name=prompt_config["nombre"]):
        
        # Loggeamos parámetros
        mlflow.log_param("modelo", "gpt-3.5-turbo")
        mlflow.log_param("pregunta", pregunta)
        mlflow.log_param("tipo_prompt", prompt_config["nombre"])
        
        # Ejecutamos
        prompt = ChatPromptTemplate.from_template(prompt_config["template"])
        chain = prompt | llm
        
        inicio = time.time()
        respuesta = chain.invoke({"pregunta": pregunta})
        tiempo = time.time() - inicio
        
        # Loggeamos métricas
        mlflow.log_metric("tiempo_respuesta", tiempo)
        mlflow.log_metric("largo_respuesta", len(respuesta.content))
        
        # Loggeamos la respuesta como artefacto
        with open("respuesta.txt", "w") as f:
            f.write(f"PROMPT: {prompt_config['template']}\n\n")
            f.write(f"RESPUESTA: {respuesta.content}")
        mlflow.log_artifact("respuesta.txt")
        
        print(f"✅ {prompt_config['nombre']}")
        print(f"   Tiempo: {tiempo:.2f}s | Largo: {len(respuesta.content)} chars")
        print(f"   Respuesta: {respuesta.content[:100]}...\n")

print("\n=== Para ver los resultados en la UI ===")
print("Corré: mlflow ui")
print("Abrí: http://localhost:5000")