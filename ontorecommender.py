#!/usr/bin/env python3
"""Provides OntoRecommender class"""

import requests
import json
import logging

__author__ = "Rafael Gonçalves, Stanford University"


class OntoRecommender:

    def __init__(self, bp_api_key):
        self.url = "http://data.bioontology.org/recommender"
        self.bp_api_key = bp_api_key
        logging.basicConfig(level=logging.INFO)

    def recommend(self, input_str, keyword_input=False):
        if keyword_input:
            input_type = 2  # 2=comma-separated list of keywords
        else:
            input_type = 1  # 1=text
            input_str = "{" + input_str + "}"

        headers = {
            "Authorization": "apiKey token=" + self.bp_api_key,
        }
        params = {
            "input": input_str,
            "input_type": input_type,
            "output_type": 1,  # 1=ranked list of ontologies
            "wc": 0.8,
            "wa": 0.0,
            "ws": 0.0,
            "wd": 0.0
        }
        response = requests.get(self.url, params=params, headers=headers, verify=True)
        if response.ok:
            json_resp = json.loads(response.content)
            if len(json_resp) > 0:
                # get ontology details
                ont = json_resp[0]['ontologies']
                ont_acr = ont[0]['acronym']
                ont_id = ont[0]['@id']

                # get coverage score and terms covered
                coverage = json_resp[0]['coverageResult']
                cov_score = coverage['score']
                cov_score_norm = coverage['normalizedScore']
                cov_terms = coverage['numberTermsCovered']
                cov_words = coverage['numberWordsCovered']

                return ont_acr, ont_id, cov_score, cov_score_norm, cov_terms, cov_words
            else:
                logging.info("Empty response from Ontology Recommender: No ontologies were found for input: " +
                             input_str + ".")
                return '', '', '', '', '', ''
        else:
            logging.error("Bad response: " + response.reason + " for URL:\n" + response.url)
