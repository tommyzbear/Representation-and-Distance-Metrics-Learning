import numpy as np
import itertools
from numba import jit
import random


@jit
def constraints_generator(train_labels, per_label_constraints=100):
    np.random.seed(1)
    distinct_labels = np.unique(train_labels)
    similar_pairs_i = []
    similar_pairs_j = []
    dissimilar_pairs_i = []
    dissimilar_pairs_j = []
    for label in distinct_labels:
        i = 0
        similar = np.where(train_labels == label)[0]
        for pair in itertools.combinations(similar, 2):
            similar_pairs_i.append(pair[0])
            similar_pairs_j.append(pair[1])
            i += 1
            if i >= per_label_constraints:
                i = 0
                break
        dissimilar = np.where(train_labels != label)[0]
        random_dissimilar_idxs = random.sample(set(dissimilar), len(dissimilar))
        for idx in random_dissimilar_idxs:
            random_current_label_idx = random.sample(set(similar), 1)[0]
            dissimilar_pairs_i.append(random_current_label_idx)
            dissimilar_pairs_j.append(idx)
            if i >= per_label_constraints:
                break

    constraints_limit = len(similar_pairs_i)
    limit_dissimilar_pairs_i = np.random.choice(dissimilar_pairs_i, constraints_limit)
    limit_dissimilar_pairs_j = np.random.choice(dissimilar_pairs_j, constraints_limit)

    #similar_dissimilar_pairs = tuple((i, j, k, l) for ((i, j), (k, l)) in zip(similar_pairs, limit_dissimilar_pairs))

    constraints = (np.asarray(similar_pairs_i),
                   np.asarray(similar_pairs_j),
                   np.asarray(limit_dissimilar_pairs_i),
                   np.asarray(limit_dissimilar_pairs_j))

    return constraints
