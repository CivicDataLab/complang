from measures import dependency_measures


def dependency_measures(graphs):
    sentences = len(graphs)
    graphs = [g for g in graphs if g is not None]
    measures = [
        dependency_based.average_average_dependency_distance,
        dependency_based.average_closeness_centrality,
        dependency_based.average_outdegree_centralization,
        dependency_based.average_closeness_centralization,
        dependency_based.average_dependents_per_word,
        dependency_based.average_longest_shortest_path,
        dependency_based.average_sentence_length,
        dependency_based.average_sentence_length_characters,
        dependency_based.average_sentence_length_syllables,
        dependency_based.average_punctuation_per_sentence,
    ]
    names = [
        "average_dependency_distance",
        "closeness_centrality",
        "outdegree_centralization",
        "closeness_centralization",
        "dependents_per_word",
        "longest_shortest_path",
        "sentence_length",
        "sentence_length_characters",
        "sentence_length_syllables",
        "punctuation_per_sentence",
    ]
    for measure, name in zip(measures, names):
        score, stdev = measure(graphs)
        print(name, score, stdev, sep="\t")
