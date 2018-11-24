#!/usr/bin/env python3
"""Provides ClusterQuality class"""

import logging
import random
import sys

from ontorecommender import OntoRecommender
from stringutils import StringUtils

__author__ = "Rafael Gonçalves, Stanford University"


class ClusterQuality:

    def __init__(self, out_file, bp_ap_key):
        self.file_writer = open(out_file, 'w')
        self.bp_ap_key = bp_ap_key
        logging.basicConfig(level=logging.INFO)

    def verify(self, cluster_dict):
        recommender = OntoRecommender(self.bp_ap_key)
        for cluster in cluster_dict:
            clust_elements = cluster_dict[cluster]
            nr_clust_elements = len(clust_elements)
            if nr_clust_elements > 150:
                clust_elements = random.sample(clust_elements, 150)
            csv_clust_elements = ",".join(clust_elements)  # comma-separated list of up to 150 cluster elements
            ont_acr, ont_id, cov_score, cov_score_norm, cov_terms, cov_words = recommender.recommend(csv_clust_elements)
            self.file_writer.write(str(cluster) + "," + str(nr_clust_elements) + ",")
            self.file_writer.write(ont_acr + "," + ont_id + "," + str(cov_score) + "," + str(cov_score_norm) + "," +
                                   str(cov_terms) + "," + str(cov_words) + "\n")
        self.file_writer.close()


if __name__ == "__main__":
    clusters_file_path = sys.argv[1]  # JSON file
    clusters_file = StringUtils.parse_cluster_dict(clusters_file_path)

    output_file = sys.argv[2]  # Output file path
    bioportal_api_key = sys.argv[3]  # BioPortal API key
    cq = ClusterQuality(output_file, bioportal_api_key)
    cq.verify(clusters_file)
