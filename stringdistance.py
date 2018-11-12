#!/usr/bin/env python3
"""Provides StringDistance class"""

import logging
import time
from enum import Enum

import jellyfish
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from similarity.jaccard import Jaccard
from stringutils import StringUtils

__author__ = "Rafael Gonçalves, Stanford University"


class Distance(Enum):
    LEVENSHTEIN = 'levenshtein'
    DAMERAU_LEVENSHTEIN = 'damerau'
    JARO = 'jaro'
    JARO_WINKLER = 'winkler'
    JACCARD = 'jaccard'
    EUCLIDEAN = 'euclidean'


class StringDistance:

    def __init__(self):
        self.vectors = dict()
        logging.basicConfig(level=logging.INFO)

    def get_levenshtein_distances(self, tokens):
        start_time = time.time()
        distances = -1 * np.array([[jellyfish.levenshtein_distance(w1, w2) for w1 in tokens] for w2 in tokens])
        end_time = time.time()
        logging.info("Levenshtein distances computation time: " + str(end_time - start_time) + "s")
        return distances

    def get_damerau_levenshtein_distances(self, tokens):
        start_time = time.time()
        distances = -1 * np.array([[jellyfish.damerau_levenshtein_distance(w1, w2) for w1 in tokens] for w2 in tokens])
        end_time = time.time()
        logging.info("Damerau-Levenshtein distances computation time: " + str(end_time - start_time) + "s")
        return distances

    def get_jaro_distances(self, tokens):
        start_time = time.time()
        distances = np.array([[jellyfish.jaro_distance(w1, w2) for w1 in tokens] for w2 in tokens])
        end_time = time.time()
        logging.info("Jaro distances computation time: " + str(end_time - start_time) + "s")
        return distances

    def get_jaro_winkler_distances(self, tokens):
        start_time = time.time()
        distances = np.array([[jellyfish.jaro_winkler(w1, w2) for w1 in tokens] for w2 in tokens])
        end_time = time.time()
        logging.info("Jaro-Winkler distances computation time: " + str(end_time - start_time) + "s")
        return distances

    def get_jaccard_distances(self, tokens):
        start_time = time.time()
        jac = Jaccard(3)
        distances = -1 * np.array([[jac.distance(w1, w2) for w1 in tokens] for w2 in tokens])
        end_time = time.time()
        logging.info("Jaccard distances computation time: " + str(end_time - start_time) + "s")
        return distances

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
        elif distance_metric == Distance.JACCARD.value:
            distances = self.get_jaccard_distances(tokens)
        # elif distance_metric == Distance.EUCLIDEAN.value:
        #     distances = self.get_euclidean_distances(tokens)
        else:
            raise ValueError("Unknown distance metric input: '" + distance_metric + "'. Supported values are: " +
                             str([distance.value for distance in Distance]))
        return distances
