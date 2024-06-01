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
from langchain_core.prompts import ChatPromptTemplate


system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Capture every single piece of information from the text without " # 1
    "sacrifing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'"
    "  - **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text. "
    "Node IDs should be exactly as they appear in the text. This means that accents, " # 2
    "diacritical marks, and capitalization must be retained in their original form. "
    'For example, if the entity in the text is "fútbol americano," it should not be altered to "Fútbol Americano" or "futbol americano". '
    "The goal is to ensure that the extracted entities are a perfect match to their appearance in the original text. \n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

GRAPH_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format and do "
                "not include any explanations. "
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)

CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided node types, relationship types and properties types in the following schema:
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

# Chain principal
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