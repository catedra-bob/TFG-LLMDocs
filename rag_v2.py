import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "q7dUtnqP"

from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
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
graph_documents = llm_transformer.convert_to_graph_documents(docs) # Prompt modificado MODIFICAR EN ESTE ARCHIVO

graph.add_graph_documents(graph_documents,
                          baseEntityLabel=True,
                          include_source=True)

graph.query("MATCH (n) SET n.id = toLower(n.id)")

# Chain principal
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Use only MATCH and RETURN statements.
The MATCH statement should have the following format, where *entity here* should be replaced with the main entity extracted from the question:
MATCH (d:Document)-[:MENTIONS]->(c:Concept *curly bracket*id: *entity here**curly bracket*)
The entity should always match letter for letter with its version provided in the schema.
For example, if the entity in the question is "Coche electrico" and the schema is "coche eléctrico", the entity in the MATCH statement should be "coche eléctrico".
The RETURN statement should contain every property of the document besides the id.

Schema:
{schema}

Notes:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Question:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True,
    validate_cypher=True,
    top_k=10,
    cypher_prompt=CYPHER_GENERATION_PROMPT
)

response = chain.invoke({"query": "¿Cuál es el sello distintivo de la cocina mediterránea?"})
print(response['result'])