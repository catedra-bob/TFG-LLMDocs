import os

os.environ["OPENAI_API_KEY"] = "sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX"

from openai import OpenAI
from langchain_community.document_loaders import DataFrameLoader
from autolabel import LabelingAgent, AutolabelDataset
from rag_v1_prompts import SYSTEM_LABEL_PROMPT, USER_LABEL_PROMPT

import pandas as pd
import json

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
    # model = OpenAI(base_url="http://openai.ull.es:8080/v1", api_key="lm-studio")
    model = OpenAI(api_key="sk-proj-S6N1LP3ePLPBDcRcU77uT3BlbkFJMsihwy3eQsyueEEIVKiX")

    i = 0
    for chunk in chunks:
        completion = model.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_LABEL_PROMPT},
                {"role": "user", "content": USER_LABEL_PROMPT + chunks[i].page_content}
            ],
            temperature=0.7
        )

        chunk.metadata['Seccion'] = completion.choices[0].message.content

        i = i + 1

    export_chunks('outputs/labeled_chunks_md.txt', chunks)

    return chunks


def export_chunks(filename, chunks):
    with open(filename, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            f.writelines(str(chunk.page_content))
            f.write("\n")
            f.writelines(str(chunk.metadata))
            f.write("\n\n")