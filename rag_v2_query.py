import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "q7dUtnqP"

from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from rag_v2_prompts import (
    GRAPH_GENERATION_PROMPT,
    CYPHER_GENERATION_PROMPT_V2,
    CYPHER_QA_PROMPT
)


graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
llm_transformer = LLMGraphTransformer(llm=llm, prompt=GRAPH_GENERATION_PROMPT)

# Chain principal
chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True,
    validate_cypher=True,
    top_k=10,
    cypher_prompt=CYPHER_GENERATION_PROMPT_V2,
    qa_prompt=CYPHER_QA_PROMPT
)

response = chain.invoke({"query": "¿Qué tiene la cocina mediterránea?"})
print(response['result'])