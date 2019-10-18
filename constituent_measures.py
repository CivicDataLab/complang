from measures import constituent_based


def constituent_measures(trees):
    sentences = len(trees)
    trees = [t for t in trees if t is not None]
    measures = [
        constituent_based.average_t_units,
        constituent_based.average_complex_t_units,
        constituent_based.average_clauses,
        constituent_based.average_dependent_clauses,
        constituent_based.average_nps,
        constituent_based.average_vps,
        constituent_based.average_pps,
        constituent_based.average_coordinate_phrases,
        constituent_based.average_constituents,
        constituent_based.average_constituents_wo_leaves,
        constituent_based.average_height,
    ]
    names = [
        "t_units",
        "complex_t_units",
        "clauses",
        "dependent_clauses",
        "nps",
        "vps",
        "pps",
        "coordinate_phrases",
        "constituents",
        "constituents_wo_leaves",
        "height",
    ]
    for measure, name in zip(measures, names):
        result = measure(trees)
        if len(result) == 4:
            print(name, result[0], result[1], sep="\t")
            print("%s_length" % name, result[2], result[3], sep="\t")
        elif len(result) == 2:
            print(name, result[0], result[1], sep="\t")
        else:
            print("two or four element needed")
