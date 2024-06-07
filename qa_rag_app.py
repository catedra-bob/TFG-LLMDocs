import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "q7dUtnqP"

from langchain_community.graphs import Neo4jGraph
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.llm import LLMChain
from langchain_community.chains.graph_qa.cypher import extract_cypher
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler
from strategy_evaluations.ragas_evaluator import ragas_evaluator, ragas_evaluator_graph
from prompts import QA_PROMPT, CYPHER_GENERATION_PROMPT_V2
from preprocess_functions import preprocess_documents

import chainlit as cl
import re


PDF_STORAGE_PATH = Path("./pdfs_otros") # Path("./pdfs_economicos") # Path("./pdfs_anaga")
COLLECTION_NAME = "coleccion_otros" # "coleccion_economicos" # "coleccion_anaga"
PROCESS = True

GENERATIVE_MODEL = ChatOpenAI(model="gpt-4o", streaming=True)
GENERATIVE_MODEL_T0 = ChatOpenAI(model_name="gpt-4o", temperature=0)


database = preprocess_documents(PDF_STORAGE_PATH, COLLECTION_NAME, PROCESS)


def rag_v1_chain():
    def format_docs(docs):
        results = ""

        for d in docs:
            results_metadatas_dict = d.metadata
            for key, value in results_metadatas_dict.items():
                regexp = re.compile(r'(Cabecera_\d)|(Punto)')
                if regexp.search(key):
                    if (key == "Punto"):
                        results += "Punto: " + value + "\n"
                    else:
                        results += str(value) + "\n"

            results += d.page_content
            results += "\n\n---\n\n"

        return results

    retriever = database.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={'score_threshold': 0.8}
                )

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | GENERATIVE_MODEL
        | StrOutputParser()
    )

    # ragas_evaluator(runnable, retriever)

    return runnable


def rag_v2_chain():
    def extract_nested_values(data):
        values = []
        for item in data:
            for value in item.values():
                if isinstance(value, dict):
                    values.extend(value.values())
                elif isinstance(value, str):
                    values.append(value)

        return values

    def retriever(question):
        use_cypher_llm_kwargs = {}
        use_cypher_llm_kwargs['prompt'] = CYPHER_GENERATION_PROMPT_V2

        cypher_generation_chain = LLMChain(
            llm=GENERATIVE_MODEL_T0,
            **use_cypher_llm_kwargs,
        )

        generated_cypher = cypher_generation_chain.run({"schema": database.schema, "question": question})
        generated_cypher = extract_cypher(generated_cypher)
        print("CYPHER: " + str(generated_cypher))
        context = database.query(generated_cypher)[: 10]
        print("CONTEXT:" + str(context))
        result = extract_nested_values(context)
        print("NODES: " + str(result))

        response = database.query("WITH " + str(result) + """AS terms 
                                MATCH (doc:Document)-[:MENTIONS]->(term)
                                WHERE term.id IN terms
                                WITH doc, terms, collect(term.id) AS mentionedTerms
                                WHERE all(term IN terms WHERE term IN mentionedTerms)
                                RETURN doc.text""")
        
        texts = [list(item.values())[0] for item in response]
        print("DOCS: " + str(texts))
        return texts

    runnable = (
        {"context": retriever, "question": RunnablePassthrough()}
        | QA_PROMPT
        | GENERATIVE_MODEL
        | StrOutputParser()
    )

    # response = rag_chain_with_source.invoke("¿Quién se enfrenta al Real Madrid?")
    # print(response['answer'])

    """def format_docs(docs):
        return "\n\n".join(doc for doc in docs)

    chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | QA_PROMPT
        | GENERATIVE_MODEL
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=chain)
    
    ragas_evaluator_graph(rag_chain_with_source)"""

    return runnable


# Inicio del pipeline RAG
@cl.on_chat_start
async def on_chat_start():
    runnable = ()

    if (isinstance(database, Chroma)):
        runnable = rag_v1_chain()
    elif (isinstance(database, Neo4jGraph)):
        runnable = rag_v2_chain()

    cl.user_session.set("runnable", runnable)


# Fases 5 y 6: Recuperación y generación
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    async with cl.Step(type="run", name="QA Assistant"):
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[
                cl.LangchainCallbackHandler(),
                PostMessageHandler(msg)
            ]),
        ):
            await msg.stream_token(chunk)

    await msg.send()