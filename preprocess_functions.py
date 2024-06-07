import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "q7dUtnqP"

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from typing import List
from langchain.docstore.document import Document
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from my_embedding_function import MyEmbeddingFunction
from prompts import GRAPH_GENERATION_PROMPT
from splitter_functions import (
    split_text_markdown,
    split_text_char,
    split_text_recursive
)
from semantic_evaluations.semantic_splitters import split_text_semantic_langchain, LLMTextSplitter
from label_functions import label_chunks_ull

import chromadb


EMBEDDINGS_MODEL = MyEmbeddingFunction()
GENERATIVE_MODEL_T0 = ChatOpenAI(model_name="gpt-4o", temperature=0)


def load_pdf(pdf_path: str):
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    return documents


def split_and_label_documents(documents: List[Document], collection_name: str):
    chunks = []

    if (collection_name == "coleccion_economicos"):
        chunks = split_text_markdown(documents, 1000, 250)
        # chunks = label_chunks_ull(chunks)
    elif (collection_name == "coleccion_anaga"):
        chunks = split_text_recursive(documents, 1000, 250)
    else:
        chunks = split_text_markdown(documents, 1000, 0)

    return chunks


def rag_v1_store_chunks_embeddings(chunks: List[Document], collection_name: str):
    chroma_client = chromadb.PersistentClient(path="./chroma")

    chroma_db = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=EMBEDDINGS_MODEL,
    )

    namespace = "chromadb/" + collection_name
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        chunks,
        record_manager,
        chroma_db,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return chroma_db


def rag_v2_store_chunks_graphs(chunks: List[Document]):
    neo4j_db = Neo4jGraph(enhanced_schema=True)
    llm_transformer = LLMGraphTransformer(llm=GENERATIVE_MODEL_T0, prompt=GRAPH_GENERATION_PROMPT)

    neo4j_db.refresh_schema()
    graph_documents = llm_transformer.convert_to_graph_documents(chunks)
    neo4j_db.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    neo4j_db.query("MATCH (n) SET n.id = toLower(n.id)")

    return neo4j_db


def preprocess_documents(pdf_storage_path: Path, collection_name: str, process: bool):
    chunks = []

    if (process):
        for pdf_path in pdf_storage_path.glob("*.pdf"):
            documents = load_pdf(str(pdf_path)) # Fase 1: Carga
            chunks += split_and_label_documents(documents, collection_name) # Fases 2 y 3: Troceado y etiquetado

    # database = rag_v1_store_chunks_embeddings(chunks, collection_name) # Fase 4: Almacenamiento como embeddings
    database = rag_v2_store_chunks_graphs(chunks) # Fase 4: Almacenamiento como grafos

    return database