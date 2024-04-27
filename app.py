from typing import List
from pathlib import Path
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    DataFrameLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler
from autolabel import LabelingAgent, AutolabelDataset
from chromadb import Documents, EmbeddingFunction, Embeddings

import chromadb
import chainlit as cl
import pandas as pd
import json

ai_client = OpenAI(base_url="http://openai.ull.es:8080/v1", api_key="lm-studio")

PDF_STORAGE_PATH = "./pdfs"


def get_embedding(text, model="CompendiumLabs/bge-large-en-v1.5-gguf"):
   global ai_client
   text = str(text).replace("\n", " ")
   emb = ai_client.embeddings.create(input = [text], model=model)
   emb = emb.data[0].embedding
   return emb


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        embed = []
        for doc in input:
            b = [float(x) for x in get_embedding(doc)]
            embed.append(b)
        return embed
    
    def embed_documents(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        embed = []
        for doc in input:
            b = [float(x) for x in get_embedding(doc)]
            embed.append(b)
        return embed
    
    def embed_query(self, input: str) -> Embeddings:
        return get_embedding(input)


mi_funcion = MyEmbeddingFunction()


def process_pdfs(pdf_storage_path: str):
    # DOCCANO
    # data = {}
    # with open('admin.jsonl') as json_data:
        # data = json.load(json_data)

    chroma_client = chromadb.PersistentClient(path="./chroma")
    pdf_directory = Path(pdf_storage_path)
    docs = []  # type: List[Document]

    if (chroma_client.count_collections() == 0):
        for pdf_path in pdf_directory.glob("*.pdf"):
            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()
            chunks_md = split_text_markdown(documents)
            chunks_md = label_chunks_ull(chunks_md)
            # chunks_md = split_text_recursive(documents)
            docs += chunks_md

        collection = chroma_client.create_collection("my_collection", embedding_function=mi_funcion)

        document_id = 0
        for chunk in docs:
            collection.add(
                metadatas=chunk.metadata,
                documents=chunk.page_content,
                ids=[str(document_id)],
            )
            document_id = document_id + 1

    doc_search = Chroma(
        client=chroma_client,
        collection_name="my_collection",
        embedding_function=mi_funcion,
    )

    namespace = "chromadb/my_documents"
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


# Divide los documentos según las etiquetas especificadas
def split_text_markdown(documents):
    all_text = ""
    for page_num in range(len(documents)):
        all_text += documents[page_num].page_content

    headers_to_split_on = [
        ("#", "Título"),
        ("##", "Capítulo"),
        ("###", "Artículo")
    ]

    text_splitter_md = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line = True
    )

    chunks_md = text_splitter_md.split_text(all_text)
    export_chunks('outputs/chunks_md.txt', chunks_md)

    chunks_char = split_text_char(chunks_md)
    export_chunks('outputs/chunks_char.txt', chunks_char)

    chunks_recursive = split_text_recursive(chunks_char)

    claves_a_anadir = ['source', 'page']
    for i in range(len(chunks_recursive)):
        for j in range(len(documents)):
            if (chunks_recursive[i].page_content.splitlines()[0] in documents[j].page_content):
                chunks_recursive[i].metadata.update({clave: documents[j].metadata[clave] for clave in claves_a_anadir if clave in documents[j].metadata})

    export_chunks('outputs/chunks_recursive.txt', chunks_recursive)

    return chunks_recursive


# Divide los documentos según los puntos que pueden haber en un artículo (1., 2., ...)
def split_text_char(documents):
    text_splitter = CharacterTextSplitter(
        separator="(\n\d+\.\s+|^\d+\.\s+)",
        chunk_size=0,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True
    )

    chunks = text_splitter.split_documents(documents)

    id = 0
    for doc in chunks:
        if (len(doc.page_content) in range(2,4)):
            if ((id + 1) < len(chunks)):
                chunks[id + 1].metadata.update({"Punto": doc.page_content})
                chunks.remove(doc)
        id += 1
    
    return chunks


# Divide los documentos según un tamaño y overlap específico
def split_text_recursive(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# Etiqueta los chunks
def label_chunks_autolabel(chunks):
    # 1. Convertir los chunks a dataframe
    df = pd.DataFrame(columns=['page_content','Título','Capítulo','Artículo','Punto'])

    i = 0
    for chunk in chunks:
        mydict = chunk.dict()

        keys = ['page_content']
        dict_page_content = {x:mydict[x] for x in keys}
        dict_metadata = mydict['metadata']

        dict_page_content.update(dict_metadata)

        df.loc[i] = dict_page_content
        i = i + 1

    df.to_excel("prueba.xlsx")

    # 2. Etiquetar el dataframe
    config = {}
    with open('config_multilabel.json') as json_data:
        config = json.load(json_data)

    agent = LabelingAgent(config)
    ds = AutolabelDataset(df, config = config)
    agent.plan(ds)
    results = agent.run(ds)

    loader = DataFrameLoader(results.df, page_content_column="page_content")
    labeled_documents = loader.load()

    # 3. Añadir las etiquetas a los chunks
    i = 0
    for chunk in chunks:
        chunk.metadata['Etiqueta'] = labeled_documents[i].metadata['labels_label']
        i = i + 1

    export_chunks('outputs/labeled_chunks_md.txt', chunks)

    return chunks


# Etiqueta los chunks
def label_chunks_ull(chunks):
    model = OpenAI(base_url="http://openai.ull.es:8080/v1", api_key="lm-studio")

    labels = [
        "Normativa Aplicable",
        "Normativa Nacional",
        "Normativa Universitaria",
        "Normativa Autonómica",
        "Estructura Presupuestaria",
        "Estructura del Documento"
    ]

    i = 0
    for chunk in chunks:
        completion = model.chat.completions.create(
        model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        messages=[
            {"role": "system", "content": "Responde sólo con las etiquetas en una lista separadas por comas."},
            {"role": "user", "content": "Eres un experto en entendiendo la normativa de la Universidad de La Laguna.\nTu trabajo es etiquetar correctamente el siguiente ejemplo con una o más de las siguientes etiquetas:\n" + str(labels) + "\nEjemplo:\n" + chunks[i].page_content}
        ],
        temperature=0.7,
        )

        chunk.metadata['Etiqueta'] = completion.choices[0].message.content

        i = i + 1

    export_chunks('outputs/labeled_chunks_md.txt', chunks)

    return chunks


def export_chunks(filename, chunks):
    with open(filename, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.writelines(str(chunk.page_content))
            f.write("\n")
            f.writelines(str(chunk.metadata))
            f.write("\n\n")


doc_search = process_pdfs(PDF_STORAGE_PATH)
model = ChatOpenAI(base_url="http://openai.ull.es:8080/v1", api_key="lm-studio", streaming=True)


@cl.on_chat_start
async def on_chat_start():
    template = """Responde en español a la pregunta basándote sólo en el siguiente contexto:

    {context}

    ---

    Responde en español a la pregunta basándote en el contexto de arriba: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
 
    def format_docs(docs):
        results = ""

        for d in docs:
            results_metadatas_dict = d.metadata
            for key, value in reversed(results_metadatas_dict.items()):
                if ((key != "page") and (key != "source") and (key != "Etiqueta")):
                    if (key == "Punto"):
                        results += "Punto: " + value + "\n"
                    else:
                        results += value + "\n"

            results += d.page_content
            results += "\n\n---\n\n"

        print(results)

        return results

    retriever = doc_search.as_retriever()

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

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