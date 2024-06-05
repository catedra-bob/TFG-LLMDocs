import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"

from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler
from my_embedding_function import MyEmbeddingFunction
from strategy_evaluations.ragas_evaluator import ragas_evaluator
from rag_v1_prompts import QA_PROMPT
from splitter_functions import (
    split_text_markdown,
    split_text_char,
    split_text_recursive,
    split_text_semantic_langchain,
    LLMTextSplitter
)
from label_functions import (
    label_chunks_autolabel,
    label_chunks_ull
)

import chromadb
import chainlit as cl
import re


PDF_STORAGE_PATH = Path("./pdfs_otros") # Path("./pdfs_economicos") # Path("./pdfs_anaga")
COLLECTION_NAME = "coleccion_otros" # "coleccion_economicos" # "coleccion_anaga"

EMBEDDINGS_MODEL = MyEmbeddingFunction()
GENERATIVE_MODEL = ChatOpenAI(model="gpt-4o", api_key="sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX", streaming=True)


def process_pdfs(pdf_storage_path: Path, collection_name: str):
    chroma_client = chromadb.PersistentClient(path="./chroma")
    docs = []

    all_collections = []
    for collection in chroma_client.list_collections():
        all_collections.append(collection.name)

    if (collection_name not in all_collections):
        # 1. Troceado y etiquetado
        for pdf_path in pdf_storage_path.glob("*.pdf"):
            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()

            if (collection_name == "coleccion_economicos"):
                chunks = split_text_markdown(documents, 1000, 250)
                # chunks = label_chunks_ull(chunks)
            elif (collection_name == "coleccion_anaga"):
                chunks = split_text_recursive(documents, 1000, 250)
            else:
                llm_splitter = LLMTextSplitter(count_tokens=True)
                chunks = llm_splitter.split_documents(documents)

            docs += chunks

        # 2. Creación de la colección
        collection = chroma_client.create_collection(collection_name, embedding_function=EMBEDDINGS_MODEL)

        document_id = 0
        for chunk in docs:
            collection.add(
                metadatas=chunk.metadata,
                documents=chunk.page_content,
                ids=[str(document_id)],
            )
            document_id = document_id + 1

    # 3. Almacenamiento
    doc_search = Chroma(
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
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search


doc_search = process_pdfs(PDF_STORAGE_PATH, COLLECTION_NAME)


# RAG pipeline
@cl.on_chat_start
async def on_chat_start():
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

    retriever = doc_search.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 3}
                )

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | GENERATIVE_MODEL
        | StrOutputParser()
    )

    ragas_evaluator(runnable, retriever)

    cl.user_session.set("runnable", runnable)


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