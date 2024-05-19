from semantic_splitters import split_text_semantic_langchain, LLMTextSplitter
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
import sys

pdf_path = "semantic_evaluations\Titulo II.pdf"
document = []
loader = PyMuPDFLoader(str(pdf_path))
document = loader.load()

flag = int(sys.argv[1])

if (flag == 0):
    with open("semantic_evaluations/pdf_langchain_chunks.txt", 'w', encoding='utf-8') as f:
        all_text = ""
        for page_num in range(len(document)):
            all_text += document[page_num].page_content

        chunks = split_text_semantic_langchain(all_text, True, 95)

        for chunk in chunks:
            f.writelines(str(chunk))
            f.write("\n---\n")
elif (flag == 1):
    with open("semantic_evaluations/pdf_gpt_chunks.txt", 'w', encoding='utf-8') as f:
        llm_splitter = LLMTextSplitter(count_tokens=True)
        chunks = llm_splitter.split_documents(document) # Le afectan los saltos de página por ser documentos

        for chunk in chunks:
            f.writelines(str(chunk))
            f.write("\n\n")
        f.write("\n---\n")