import sys

sys.path.append('..')

from dotenv import load_dotenv
from langchain.text_splitter import TextSplitter
from typing import List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from app_chroma.my_embedding_function import MyEmbeddingFunction
from app_chroma.semantic_evaluations.semantic_representations import represent_chunks
from langchain_experimental.text_splitter import SemanticChunker
from app_chroma.rag_v1_prompts import LLM_SPLITTER_PROMPT
import re
import tiktoken 

load_dotenv()

# Método Langchain semantic chunker
def split_text_semantic_langchain(text, represent, treshold):
    text_splitter = SemanticChunker(MyEmbeddingFunction(), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=treshold)
    chunks_semantic = text_splitter.split_text(text)
    if (represent): represent_chunks(text_splitter, text)

    return chunks_semantic


def split_documents_semantic_langchain(documents, represent, treshold):
    text_splitter = SemanticChunker(MyEmbeddingFunction(), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=treshold)
    chunks_semantic = text_splitter.split_documents(documents)
    if (represent):
        all_text = ""
        for page_num in range(len(documents)):
            all_text += documents[page_num].page_content
        represent_chunks(text_splitter, all_text)

    return chunks_semantic


# Clase ChatGPT (gpt-4o)
class LLMTextSplitter(TextSplitter):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        count_tokens: bool = False,
        encoding_name: str = "cl100k_base",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name 
        self.count_tokens = count_tokens
        self.encoding_name = encoding_name
        self.model = ChatOpenAI(temperature=0, model=self.model_name, api_key="sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX")
        self.output_paser = StrOutputParser() 

        topic_template = LLM_SPLITTER_PROMPT
        self.prompt_template = ChatPromptTemplate.from_template(topic_template)
        self.chain = self.prompt_template | self.model | self.output_paser

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(string) )
        return num_tokens

    def split_text (self, text: str) -> List[str]:
        if self.count_tokens:
            token_count = self.num_tokens_from_string(text)
            print(f"Token count of input text: {token_count}")
        
        response = self.chain.invoke({"text": text})
        return self._format_chunks (response)
    
    def _format_chunks(self, text: str) -> List[str]:
        pattern = r">>>(.*?)<<<"
        chunks = re.findall(pattern, text, re.DOTALL)
        formatted_chunks = [chunk.strip() for chunk in chunks]
        return formatted_chunks