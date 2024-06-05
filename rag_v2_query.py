import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "q7dUtnqP"

from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from strategy_evaluations.ragas_evaluator import ragas_evaluator_graph
from rag_v2_prompts import (
    CYPHER_GENERATION_PROMPT_V2,
    CYPHER_QA_PROMPT
)


graph = Neo4jGraph()
llm_qa = ChatOpenAI(model_name="gpt-4o")
llm_cypher = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Chain principal
chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=llm_cypher,
    qa_llm=llm_qa,
    verbose=True,
    validate_cypher=True,
    top_k=10,
    cypher_prompt=CYPHER_GENERATION_PROMPT_V2,
    qa_prompt=CYPHER_QA_PROMPT
)

# ragas_evaluator_graph(chain)

response = chain.invoke({"query": "¿Cuál es el sello distintivo de la cocina mediterránea?"})
print(response['result'])
chain.cypher_generation_chain

# with open("schema.txt", 'w', encoding='utf-8') as f:
    # f.writelines(str(graph.schema))