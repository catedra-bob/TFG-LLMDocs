from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

def ragas_evaluator(runnable, retriever):
    testset = pd.read_excel("test_set.xlsx")

    questions = testset["question"].to_list()
    ground_truth = testset["ground_truth"].to_list()

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}

    for query in questions:
        data["question"].append(query)
        data["answer"].append(runnable.invoke(query))
        data["contexts"].append([doc.page_content for doc in retriever.get_relevant_documents(query)])

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset = dataset,
        metrics=[
            context_relevancy,
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    df = result.to_pandas()
    df.to_csv("result.csv")

    heatmap_data = df[['context_relevancy', 'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']]

    cmap = LinearSegmentedColormap.from_list('green_red', ['red', 'green'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5, cmap=cmap)

    plt.yticks(ticks=range(len(df['question'])), labels=df['question'], rotation=0)

    plt.show()