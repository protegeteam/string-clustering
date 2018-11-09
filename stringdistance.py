#!/usr/bin/env python3
"""Provides StringDistance class"""

from enum import Enum

import jellyfish
import numpy as np
import spacy

__author__ = "Rafael Gonçalves, Stanford University"


class Distance(Enum):
    LEVENSHTEIN = 'levenshtein'
    DAMERAU_LEVENSHTEIN = 'damerau'
    JARO = 'jaro'
    JARO_WINKLER = 'winkler'
    MATCH_RATING = 'match_rating'
    EUCLIDEAN = 'euclidean'


class StringDistance:

    def __init__(self):
        self.vectors = dict()

    def get_levenshtein_distances(self, tokens):
        distances = -1 * np.array([[jellyfish.levenshtein_distance(w1, w2) for w1 in tokens] for w2 in tokens])
        return distances

    def get_damerau_levenshtein_distances(self, tokens):
        distances = -1 * np.array([[jellyfish.damerau_levenshtein_distance(w1, w2) for w1 in tokens] for w2 in tokens])
        return distances

    def get_jaro_distances(self, tokens):
        distances = -1 * np.array([[jellyfish.jaro_distance(w1, w2) for w1 in tokens] for w2 in tokens])
        return distances

    def get_jaro_winkler_distances(self, tokens):
        distances = -1 * np.array([[jellyfish.jaro_winkler(w1, w2) for w1 in tokens] for w2 in tokens])
        return distances

    def get_match_rating_distances(self, tokens):
        distances = -1 * np.array([[jellyfish.match_rating_comparison(w1, w2) for w1 in tokens] for w2 in tokens])
        return distances

    def get_euclidean_distances(self, tokens):
        vectors = self.get_vectors(tokens).values()
        return np.array([[t1.similarity(t2) for t1 in vectors] for t2 in vectors])

    # get a dictionary of vectors for the given tokens
    def get_vectors(self, tokens):
        # load spacy statistical model. we are interested in the word vectors; 300-dimensional vector representations
        # of words that allow us to determine how similar they are to each other.
        model = spacy.load('en_core_web_lg')
        vectors = dict()
        for token in tokens:
            vectors[token] = self.get_vector(token, model)
        return vectors

    # get vector of the given token
    def get_vector(self, token, model):
        if token not in self.vectors:
            vector = model(str(token))
            self.vectors[token] = vector
            return vector
        else:
            return self.vectors[token]

    # takes a collection of tokens and computes the pairwise distance between all tokens,
    # according to the specified distance metric
    def get_distances(self, tokens, distance_metric):
        if distance_metric == Distance.LEVENSHTEIN.value:
            distances = self.get_levenshtein_distances(tokens)
        elif distance_metric == Distance.DAMERAU_LEVENSHTEIN.value:
            distances = self.get_damerau_levenshtein_distances(tokens)
        elif distance_metric == Distance.JARO.value:
            distances = self.get_jaro_distances(tokens)
        elif distance_metric == Distance.JARO_WINKLER.value:
            distances = self.get_jaro_winkler_distances(tokens)
        elif distance_metric == Distance.MATCH_RATING.value:
            distances = self.get_match_rating_distances(tokens)
        elif distance_metric == Distance.EUCLIDEAN.value:
            distances = self.get_euclidean_distances(tokens)
        else:
            raise ValueError("Unknown distance metric input: '" + distance_metric + "'. Supported values are: " +
                             str([distance.value for distance in Distance]))
        return distances
