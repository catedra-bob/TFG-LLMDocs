from openai import OpenAI
from chromadb import Documents, EmbeddingFunction, Embeddings

ai_client = OpenAI(base_url="http://openai.ull.es:8080/v1", api_key="lm-studio")

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