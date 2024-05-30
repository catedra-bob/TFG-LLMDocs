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


graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
llm_transformer = LLMGraphTransformer(llm=llm)

docs = []
for pdf_path in Path("./pdfs_otros").glob("*.pdf"):
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    chunks = split_text_markdown(documents, 1000, 0)
    chunks.append(Document(page_content="Los alimentos de la cocina se trasladan en vehículos eléctricos", metadata={"source": "local"}))
    docs += chunks

graph_documents = llm_transformer.convert_to_graph_documents(docs)

graph.add_graph_documents(graph_documents,
                          baseEntityLabel=True,
                          include_source=True)

# QUERY

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="Todas las entidades que aparecen en la pregunta"
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Estás extrayendo todas las entidades de la pregunta."
        ),
        (
            "human",
            "Extrae todas las entidades de la siguiente pregunta: {question}"
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)
print(entity_chain.invoke({"question": "¿Qué tiene la cocina mediterránea?"}).names)

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

structured_data = structured_retriever("¿Qué tiene la cocina mediterránea?")
print(structured_data)