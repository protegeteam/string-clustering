#!/usr/bin/env python3
"""Provides StringClusters class.

StringClusters is a tool for comparing strings using different distance metrics, and for clustering strings using
various clustering algorithms according to the pairwise distances between them.

Example usage:
    ```
    Cluster the list of strings in a file named "foods.txt" in the current directory:
    >> python stringclusters.py -i foods.txt

    Cluster the list of strings in a file named "foods.txt" and save the clusters to a file named "clusters.json":
    >> python stringclusters.py -i foods.txt -o /home/user/documents/clusters.json

    Cluster the list of strings in a file named "foods.txt" using Euclidean distance:
    >> python stringclusters.py -i foods.txt -d euclidean
    ```
"""

import argparse
import datetime
import logging
import os
import sys
import time
from enum import Enum

import hdbscan
import numpy as np
import sklearn.cluster
from stringdistance import Distance, StringDistance
from stringnormalize import StringNormalize
from stringutils import StringUtils

__author__ = "Rafael Gonçalves, Stanford University"


class Algorithm(Enum):
    AFFINITY_PROPAGATION = 'ap'
    DBSCAN = 'dbscan'
    HDBSCAN = 'hdbscan'
    MEAN_SHIFT = 'ms'


class StringClusters:

    def __init__(self, output_folder):
        self.output_folder = output_folder
        logging.basicConfig(level=logging.INFO)

    # cluster the given tokens according to their similarity distances using affinity propagation.
    # returns a dictionary that maps each cluster exemplar to an array of cluster elements (incl. exemplar)
    def cluster_affinity_propagation(self, distances, tokens):
        start_time = time.time()

        ap = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.8)
        ap.fit(-1 * distances)  # input to affinity propagation is an array of similarities

        end_time = time.time()
        logging.info("Affinity propagation clustering time: " + str(end_time - start_time) + "s")
        return self.build_ap_cluster_dictionary(ap.labels_, tokens, ap.cluster_centers_indices_)

    # HDBSCAN clustering
    def cluster_hdbscan(self, distances, tokens):
        start_time = time.time()

        hdbscan_ = hdbscan.HDBSCAN(min_samples=5, alpha=1.0, min_cluster_size=2, metric='precomputed')
        hdbscan_.fit(distances.astype(np.float64))

        end_time = time.time()
        logging.info("HDBSCAN clustering time: " + str(end_time-start_time) + "s")
        return self.build_cluster_dictionary(hdbscan_.labels_, tokens)

    # DBSCAN clustering
    def cluster_dbscan(self, distances, tokens, distance):
        start_time = time.time()

        eps = self.get_eps_dbscan(distance)
        dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        dbscan.fit(distances)  # input is an array of distances

        end_time = time.time()
        logging.info("DBSCAN clustering time: " + str(end_time - start_time) + "s")
        return self.build_cluster_dictionary(dbscan.labels_, tokens)

    def get_eps_dbscan(self, distance):
        # eps of ~16.0 works well for jaro(-winkler) distances
        if distance == Distance.JARO.value or distance == Distance.JARO_WINKLER.value:
            eps = 16.0
        # eps of ~40.0 works well for jaccard and cosine distances
        elif distance == Distance.JACCARD.value or distance == Distance.COSINE.value:
            eps = 40.0
        # eps of ~3.0 works well for levenshtein(-damerau) distances
        elif distance == Distance.LEVENSHTEIN.value or distance == Distance.DAMERAU_LEVENSHTEIN.value:
            eps = 3.0
        else:
            eps = 0.5
        return eps

    # MeanShift clustering
    def cluster_meanshift(self, distances, tokens):
        start_time = time.time()

        bandwidth = sklearn.cluster.estimate_bandwidth(distances, quantile=0.2, n_samples=50)
        # bandwith of ~140.0 works well for all distance metrics but levenshtein(-damerau)
        # bandwith of ~23.0 works well for levenshtein(-damerau)
        meanshift = sklearn.cluster.MeanShift(bandwidth=bandwidth, cluster_all=True)
        meanshift.fit(distances)

        end_time = time.time()
        logging.info("Mean shift clustering time: " + str(end_time - start_time) + "s")
        return self.build_cluster_dictionary(meanshift.labels_, tokens)

    def build_cluster_dictionary(self, labels, tokens):
        clusters = dict()
        key = 0
        for cluster_id in np.unique(labels):
            cluster = np.unique(tokens[np.nonzero(labels == cluster_id)])
            clusters[key] = cluster.tolist()
            key += 1
        return clusters

    def build_ap_cluster_dictionary(self, labels, tokens, centers_indices):
        clusters = dict()
        for cluster_id in np.unique(labels):
            exemplar = tokens[centers_indices[cluster_id]]
            cluster = np.unique(tokens[np.nonzero(labels == cluster_id)])
            clusters[exemplar] = cluster.tolist()
        return clusters

    def cluster(self, tokens, distance_metric, clustering_algorithm, ngrams):
        tokens = np.array(list(StringNormalize().normalize_tokens(tokens)))
        distances = StringDistance().get_distances(tokens, distance_metric, ngrams)

        if clustering_algorithm == Algorithm.AFFINITY_PROPAGATION.value:
            clusters = self.cluster_affinity_propagation(distances, tokens)
        elif clustering_algorithm == Algorithm.DBSCAN.value:
            clusters = self.cluster_dbscan(distances, tokens, distance_metric)
        elif clustering_algorithm == Algorithm.HDBSCAN.value:
            clusters = self.cluster_hdbscan(distances, tokens)
        elif clustering_algorithm == Algorithm.MEAN_SHIFT.value:
            clusters = self.cluster_meanshift(distances, tokens)
        else:
            raise ValueError("Unknown clustering algorithm: '" + clustering_algorithm + "'. Supported values are: " +
                             str([alg.value for alg in Algorithm]))

        StringUtils.save_dictionary_as_json(self.output_folder + "clusters_" + clustering_algorithm + "_" +
                                            distance_metric + ".json", clusters)
        StringUtils.save_distances(self.output_folder + "distances_" + distance_metric + ".csv", distances, tokens)
        return clusters


# Use arparse to get command line arguments
def get_arguments():
    # get timestamp in ISO format, and replace colons with dashes in timestamp to have a valid file name
    timestamp = datetime.datetime.now().isoformat().replace(":", "-")
    default_output_file = "stringclusters_output_" + timestamp + ".json"
    parser = argparse.ArgumentParser(
        description="StringClusters is a tool to compare strings using different distance metrics, and to cluster them "
                    "using various clustering algorithms according to the pairwise distances between strings.")
    parser.add_argument("-i", "--input_file", required=True, type=str,
                        help="Input file containing list of strings (one per line)")
    parser.add_argument("-o", "--output_file", required=False, type=str, default=default_output_file,
                        help="Output file. By default saves as 'stringclusters_output.json' with a creation timestamp, "
                             "to the current directory")
    parser.add_argument("-d", "--distance_metric", required=False, type=str, default=Distance.LEVENSHTEIN,
                        help="Distance metric (levenshtein | damerau | jaro | winkler | match_rating | euclidean). "
                             "Default: Levenshtein distance ('levenshtein')")
    parser.add_argument("-c", "--clustering", required=False, type=str, default=Algorithm.AFFINITY_PROPAGATION,
                        help="Clustering algorithm (ap | ms | dbscan | hbscan). "
                             "Supported algorithms are: affinity propagation (ap), mean shift (ms), DBSCAN (dbscan),"
                             "and HDBSCAN (hdbscan). Default: affinity propagation ('ap')")
    parser.add_argument("-n", "--ngrams", required=False, type=int, default=4,
                        help="Number of characters 'n' for n-grams based algorithms, which work by converting strings "
                             "into sets of n-grams (sequences of n characters). Default: 4.")
    arguments = parser.parse_args()

    if not os.path.exists(arguments.input_file):
        parser.error("The file '{}' does not exist".format(arguments.input_file))
        sys.exit(1)

    # create output directories if needed
    if os.path.dirname(arguments.output_file):
        os.makedirs(os.path.dirname(arguments.output_file), exist_ok=True)

    return arguments.input_file, arguments.output_file, arguments.distance_metric, arguments.clustering, arguments.ngrams


if __name__ == "__main__":
    args = get_arguments()
    strings = StringUtils.parse_file(args[0])
    StringClusters(args[1]).cluster(strings, args[2], args[3], args[4])
