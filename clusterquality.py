#!/usr/bin/env python3
"""Provides ClusterQuality class"""

import logging
import random
import sys
from distutils import util
from ontorecommender import OntoRecommender
from stringutils import StringUtils

__author__ = "Rafael Gonçalves, Stanford University"


class ClusterQuality:

    def __init__(self, out_file, bp_ap_key):
        self.file_writer = open(out_file, 'w')
        self.bp_ap_key = bp_ap_key
        logging.basicConfig(level=logging.DEBUG)

    def verify(self, cluster_dict, keyword_input=False):
        recommender = OntoRecommender(self.bp_ap_key)
        for cluster in cluster_dict:
            clust_elements = cluster_dict[cluster]
            nr_clust_elements = len(clust_elements)
            if not keyword_input:
                clust_elements = self.tokenize_str_array(clust_elements)

            # take a sample of up to 150 cluster elements to ensure that BioPortal handles the request
            if nr_clust_elements > 150:
                clust_elements = random.sample(clust_elements, 150)

            if keyword_input:
                csv_clust_elements = ",".join(clust_elements)
            else:
                csv_clust_elements = " ".join(clust_elements)

            ont_acr, ont_id, cov_score, cov_score_norm, cov_terms, cov_words = recommender.recommend(csv_clust_elements, keyword_input)
            self.file_writer.write(str(cluster) + "," + str(nr_clust_elements) + ",")
            self.file_writer.write(ont_acr + "," + ont_id + "," + str(cov_score) + "," + str(cov_score_norm) + "," +
                                   str(cov_terms) + "," + str(cov_words) + "\n")
        self.file_writer.close()

    def tokenize_str_array(self, array):
        output = []
        for element in array:
            tokens = element.split()
            for token in tokens:
                output.append(token)
        return output


if __name__ == "__main__":
    clusters_file_path = sys.argv[1]  # JSON file
    clusters_file = StringUtils.parse_cluster_dict(clusters_file_path)

    output_file = sys.argv[2]  # Output file path
    bioportal_api_key = sys.argv[3]  # BioPortal API key
    cq = ClusterQuality(output_file, bioportal_api_key)

    use_keyword_input = bool(util.strtobool(sys.argv[4]))  # True=keyword-based input, False=Raw text input
    cq.verify(clusters_file, keyword_input=use_keyword_input)
