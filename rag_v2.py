import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "q7dUtnqP"

from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from splitter_functions import (
    split_text_markdown
)
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.docstore.document import Document
from langchain_core.prompts.prompt import PromptTemplate


graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
llm_transformer = LLMGraphTransformer(llm=llm)

# Creación de los chunks
docs = []
for pdf_path in Path("./pdfs_otros").glob("*.pdf"):
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    chunks = split_text_markdown(documents, 1000, 0)
    chunks.append(Document(page_content="Los alimentos de la cocina se trasladan en vehículos eléctricos", metadata={"source": "local"}))
    docs += chunks

# Creación del la base de datos orientada a grafos
graph_documents = llm_transformer.convert_to_graph_documents(docs) # Prompt modificado

graph.add_graph_documents(graph_documents,
                          baseEntityLabel=True,
                          include_source=True)

graph.query("MATCH (n) SET n.id = toLower(n.id)")

# Chain principal
QUESTION = "¿Cuál es el sello distintivo de la cocina mediterránea?"

CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
prompt.format(schema=str(graph.schema), question=QUESTION)

chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True,
    validate_cypher=True,
    top_k=10,
    cypher_prompt=prompt
)

response = chain.invoke({"query": QUESTION})
print(response['result'])