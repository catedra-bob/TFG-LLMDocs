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


def ragas_evaluator(runnable, retriever = 0):
    testset = pd.read_excel("strategy_evaluations/economicos/my_economicos_test_set.xlsx")
    # testset = pd.read_excel("strategy_evaluations/temas_variados/temas_variados_test_set.xlsx")

    questions = testset["question"].to_list()
    ground_truth = testset["ground_truth"].to_list()

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}

    if (retriever != 0): # RAG v1
        for query in questions:
            data["question"].append(query)
            data["answer"].append(runnable.invoke(query))
            data["contexts"].append([doc.page_content for doc in retriever.get_relevant_documents(query)])
    else: # RAG v2
        for query in questions:
            response = runnable.invoke(query)
            data["question"].append(query)
            data["answer"].append(response['answer'])
            data["contexts"].append(response['context'])

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

    plot_results(result)


def plot_results(result):
    df = result.to_pandas()
    df.to_csv("strategy_evaluations/economicos/rag_v1/result.csv")
    # df.to_csv("strategy_evaluations/temas_variados/result.csv")

    metrics_df = df[['context_relevancy', 'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']]
    metrics_df.loc['total'] = metrics_df.mean()
    media_total = metrics_df.loc['total'].mean()
    media_total = "{:.2f}".format(media_total)
    print("Puntuación final: " + str(media_total))

    cmap = LinearSegmentedColormap.from_list('green_red', ['red', 'green'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df, annot=True, fmt=".2f", linewidths=.5, cmap=cmap)

    plt.yticks(ticks=range(len(df['question'])), labels=df['question'], rotation=0)

    plt.show()