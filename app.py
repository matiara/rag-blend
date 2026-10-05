from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader

load_dotenv()

st.set_page_config(page_title="Blend AI Assistant", page_icon="🤖")
st.title("🤖 Blend AI Assistant")
st.caption("Sistema RAG construido con LangChain + OpenAI")

@st.cache_resource
def cargar_rag():
    loader = TextLoader("blend_info.txt", encoding="utf-8")
    documentos = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    fragmentos = splitter.split_documents(documentos)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(fragmentos, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template("""
Sos un asistente experto en Blend. Respondé de forma clara y concisa 
basándote solo en el siguiente contexto. Si no sabés algo, decilo.

Contexto:
{context}

Pregunta: {question}

Respuesta:
""")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain

# Inicializar historial
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Cargar RAG
with st.spinner("Cargando sistema RAG..."):
    chain = cargar_rag()

# Mostrar historial
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["contenido"])

# Input del usuario
if pregunta := st.chat_input("Preguntame algo sobre Blend..."):
    # Mostrar pregunta
    with st.chat_message("user"):
        st.write(pregunta)
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta = chain.invoke(pregunta)
            st.write(respuesta.content)
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta.content})