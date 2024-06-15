import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = ""

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
from split_functions import (
    split_documents_markdown,
    split_documents_recursive,
    export_chunks
)
from semantic_evaluations.semantic_splitters import (
    split_documents_semantic_langchain,
    LLMTextSplitter
)
from label_functions import label_chunks_llm

import chromadb


EMBEDDINGS_MODEL = OpenAIEmbeddings()
GENERATIVE_MODEL_T0 = ChatOpenAI(model_name="gpt-4o", temperature=0)


def load_pdf(pdf_path: str):
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    return documents


def split_and_label_documents(documents: List[Document], collection_name: str):
    chunks = []

    if (collection_name == "coleccion_economicos"):
        # chunks = split_documents_recursive(documents, 1000, 250)

        chunks = split_documents_markdown(documents, 1000, 250)

        # chunks = split_documents_semantic_langchain(documents, False, 30)

        # llm_splitter = LLMTextSplitter(count_tokens=True)
        # chunks = llm_splitter.split_documents(documents)
        # export_chunks('outputs/chunks_semantic_gpt.txt', chunks)

        # chunks = label_chunks_llm(chunks)
    elif (collection_name == "coleccion_anaga"):
        chunks = split_documents_recursive(documents, 1000, 250)
    else:
        chunks = split_documents_recursive(documents, 250, 50)

        # chunks = split_documents_markdown(documents, 250, 50)

        # chunks = split_documents_semantic_langchain(documents, False, 30)

        # llm_splitter = LLMTextSplitter(count_tokens=True)
        # chunks = llm_splitter.split_documents(documents)
        # export_chunks('outputs/chunks_semantic_gpt.txt', chunks)

    return chunks


def rag_v1_store_embeddings(chunks: List[Document], collection_name: str):
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


def rag_v2_store_graphs(chunks: List[Document]):
    # Creación del grafo
    llm_transformer = LLMGraphTransformer(llm=GENERATIVE_MODEL_T0, prompt=GRAPH_GENERATION_PROMPT)
    graph_documents = llm_transformer.convert_to_graph_documents(chunks)

    # Almacenamiento del grafo
    neo4j_db = Neo4jGraph(enhanced_schema=True)

    neo4j_db.query("MATCH (n) DETACH DELETE n")
    neo4j_db.refresh_schema()
    neo4j_db.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    neo4j_db.query("MATCH (n) SET n.id = toLower(n.id)")


def preprocess_documents(pdf_storage_path: Path, collection_name: str, rag_version: int):
    chunks = []

    for pdf_path in pdf_storage_path.glob("*.pdf"):
        documents = load_pdf(str(pdf_path)) # Fase 1: Carga
        chunks += split_and_label_documents(documents, collection_name) # Fases 2 y 3: Troceado y etiquetado

    if (rag_version == 1):
        rag_v1_store_embeddings(chunks, collection_name) # Fase 4: Almacenamiento como embeddings
    elif (rag_version == 2):
        rag_v2_store_graphs(chunks) # Fase 4: Almacenamiento como grafos
