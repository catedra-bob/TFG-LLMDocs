import sys

sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import re

def represent_chunks(text_splitter, documents):
    all_text = ""
    for page_num in range(len(documents)):
        all_text += documents[page_num].page_content

    # Splitting the text on '.', '?', and '!'
    single_sentences_list = re.split(r'(?<=[.?!])\s+', all_text)

    distances, sentences = text_splitter._calculate_sentence_distances(single_sentences_list)

    plt.plot(distances)

    y_upper_bound = .25
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))

    # We need to get the distance threshold that we'll consider an outlier
    breakpoint_distance_threshold = text_splitter._calculate_breakpoint_threshold(distances)

    plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')

    # Then we'll see how many distances are actually above this one
    num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold]) # The amount of distances above your threshold
    plt.text(x=(len(distances)*.01), y=y_upper_bound/50, s=f"{num_distances_above_theshold + 1} Chunks")

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list

    # Start of the shading and text
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, breakpoint_index in enumerate(indices_above_thresh):
        start_index = 0 if i == 0 else indices_above_thresh[i - 1]
        end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)

        plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
        plt.text(x=np.average([start_index, end_index]),
                y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
                s=f"Chunk #{i}", horizontalalignment='center',
                rotation='vertical')

    # # Additional step to shade from the last breakpoint to the end of the dataset
    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.25)
            plt.text(x=np.average([last_breakpoint, len(distances)]),
                    y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
                    s=f"Chunk #{i+1}",
                    rotation='vertical')

    plt.title("Chunks Based On Embedding Breakpoints")
    plt.xlabel("Index of sentences in essay (Sentence Position)")
    plt.ylabel("Cosine distance between sequential sentences")
    plt.show()