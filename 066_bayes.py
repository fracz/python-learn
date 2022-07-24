from collections import defaultdict

import numpy as np

X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0],
])

Y_train = ['Y', 'N', 'Y', 'Y']

X_test = np.array([
    [1, 1, 0],
])


def get_label_indices(labels):
    indices = defaultdict(list)
    for index, label in enumerate(labels):
        indices[label].append(index)
    return indices


label_indices = get_label_indices(Y_train)
print(label_indices)


def get_prior(label_indices):
    # return {'Y': .5, 'N': .5}
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior


prior = get_prior(label_indices)
print(prior)


def get_likelihood(features, label_indices, smoothing = 0):
    likelihood = {}
    feature_possible_values = 2
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + feature_possible_values * smoothing)
    return likelihood


likelihood = get_likelihood(X_train, label_indices, smoothing=1)
print(likelihood)


def get_posterior(X, prior, likelihood):
    posteriors = []
    for x in X:
        posterior = prior.copy()
        for label, label_likelihoods in likelihood.items():
            for index, value in enumerate(x):
                posterior[label] *= label_likelihoods[index] if value else 1 - label_likelihoods[index]
        sum_posterior = sum(posterior.values())
        for label in posterior:
            posterior[label] /= sum_posterior
        posteriors.append(posterior)
    return posteriors


result = get_posterior(X_test, prior, likelihood)
print(result)