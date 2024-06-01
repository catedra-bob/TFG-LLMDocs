import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "q7dUtnqP"

from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from splitter_functions import (
    split_text_markdown
)
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.docstore.document import Document
from rag_v2_prompts import (
    GRAPH_GENERATION_PROMPT
)


graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
llm_transformer = LLMGraphTransformer(llm=llm, prompt=GRAPH_GENERATION_PROMPT)

# Creación de los chunks
docs = []
for pdf_path in Path("./pdfs_otros").glob("*.pdf"):
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    chunks = split_text_markdown(documents, 1000, 0)
    chunks.append(Document(page_content="Los alimentos de la cocina se trasladan en vehículos eléctricos", metadata={"source": "local"}))
    docs += chunks

# Creación del la base de datos orientada a grafos
graph_documents = llm_transformer.convert_to_graph_documents(docs)

graph.add_graph_documents(graph_documents,
                          baseEntityLabel=True,
                          include_source=True)

graph.query("MATCH (n) SET n.id = toLower(n.id)")