from semantic_splitters import LLMTextSplitter, split_text_semantic_langchain_graph
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

document = []
pdf_directory = Path("./semantic_evaluations")
for pdf_path in pdf_directory.glob("*.pdf"):
    print(str(pdf_path))
    loader = PyMuPDFLoader(str(pdf_path))
    document = loader.load()

with open("semantic_evaluations/pdf_langchain_chunks.txt", 'a', encoding='utf-8') as f:
    all_text = ""
    for page_num in range(len(document)):
        all_text += document[page_num].page_content

    chunks = split_text_semantic_langchain_graph(all_text)

    for chunk in chunks:
        f.writelines(str(chunk))
        f.write("\n---\n")

with open("semantic_evaluations/pdf_gpt_chunks.txt", 'a', encoding='utf-8') as f:
    llm_splitter = LLMTextSplitter(count_tokens=True)
    chunks = llm_splitter.split_documents(document)

    for chunk in chunks:
        f.writelines(str(chunk))
        f.write("\n\n")
    f.write("\n---\n")