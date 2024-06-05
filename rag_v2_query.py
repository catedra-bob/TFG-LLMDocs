import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "q7dUtnqP"

from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.chains.graph_qa.cypher import extract_cypher
from strategy_evaluations.ragas_evaluator import ragas_evaluator_graph
from rag_v2_prompts import (
    CYPHER_GENERATION_PROMPT_V2,
    CYPHER_QA_PROMPT
)


graph = Neo4jGraph()
llm_qa = ChatOpenAI(model_name="gpt-4o")
llm_cypher = ChatOpenAI(temperature=0, model_name="gpt-4o")

# v2
"""chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=llm_cypher,
    qa_llm=llm_qa,
    verbose=True,
    validate_cypher=True,
    top_k=10,
    cypher_prompt=CYPHER_GENERATION_PROMPT_V2,
    qa_prompt=CYPHER_QA_PROMPT
)

response = chain.invoke({"query": "¿Cuál es el sello distintivo de la cocina mediterránea?"})
print(response['result'])"""

# v3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.llm import LLMChain

def extract_nested_values(data):
    values = []
    for item in data:
        for value in item.values():
            if isinstance(value, dict):
                values.extend(value.values())
    return values

def retriever(question):
    use_cypher_llm_kwargs = {}
    use_cypher_llm_kwargs['prompt'] = CYPHER_GENERATION_PROMPT_V2

    cypher_generation_chain = LLMChain(
        llm=llm_cypher,
        **use_cypher_llm_kwargs,
    )

    generated_cypher = cypher_generation_chain.run({"schema": graph.schema, "question": question})
    generated_cypher = extract_cypher(generated_cypher)
    print("CYPHER: " + str(generated_cypher))
    context = graph.query(generated_cypher)[: 10]
    result = extract_nested_values(context)
    print("NODES: " + str(result))

    response = graph.query("WITH " + str(result) + """AS terms 
                            MATCH (doc:Document)-[:MENTIONS]->(term)
                            WHERE term.id IN terms
                            WITH doc, terms, collect(term.id) AS mentionedTerms
                            WHERE all(term IN terms WHERE term IN mentionedTerms)
                            RETURN doc.text""")
    
    texts = [list(item.values())[0] for item in response]
    print("DOCS: " + str(texts))
    return texts

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

qa_chain = (
    RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm_qa
    | StrOutputParser()
)


ragas_evaluator_graph(qa_chain, retriever)

# with open("schema.txt", 'w', encoding='utf-8') as f:
    # f.writelines(str(graph.schema))