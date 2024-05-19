import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from semantic_evaluations.semantic_splitters import split_text_semantic_langchain, LLMTextSplitter

# Divide los documentos según las etiquetas markdown que contiene
def split_text_markdown(documents):
    all_text = ""
    for page_num in range(len(documents)):
        all_text += documents[page_num].page_content

    headers_to_split_on = [
        ("#", "Cabecera 1"),
        ("##", "Cabecera 2"),
        ("###", "Cabecera 3"),
        ("####", "Cabecera 4"),
        ("#####", "Cabecera 5"),
        ("######", "Cabecera 6"),
    ]

    text_splitter_md = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line = True
    )

    chunks_md = text_splitter_md.split_text(all_text)
    export_chunks('outputs/chunks_md.txt', chunks_md)

    chunks = []

    if (documents[0].metadata["source"] == "pdfs_economicos\Bases Ejecución 2024 (1).pdf"):
        chunks = split_text_char(chunks_md)
        export_chunks('outputs/chunks_char.txt', chunks)

    chunks = split_text_recursive(chunks, 1000, 250)
    chunks = add_source(chunks, documents)
    export_chunks('outputs/chunks_recursive.txt', chunks)

    return chunks


# Divide los documentos según los puntos que pueden haber en un apartado (1., 2., ...)
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
def split_text_recursive(documents, size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def add_source(chunks, documents):
    claves_a_anadir = ['source', 'page']
    for i in range(len(chunks)):
        chunks[i].metadata.update({'source': documents[0].metadata['source']})
        flag = 0

        for j in range(len(documents)):
            if (chunks[i].page_content in documents[j].page_content):
                chunks[i].metadata.update({'page': documents[j].metadata['page']})
                flag = 1

        if (flag == 0):
            for j in range(len(documents)):
                if (chunks[i].page_content.splitlines()[0] in documents[j].page_content):
                    chunks[i].metadata.update({'page': documents[j].metadata['page']})
    
    return chunks


def export_chunks(filename, chunks):
    with open(filename, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            f.writelines(str(chunk.page_content))
            f.write("\n")
            f.writelines(str(chunk.metadata))
            f.write("\n\n")