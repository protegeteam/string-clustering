#!/usr/bin/env python3
"""Provides TextAnnotator class"""

import sys
import time

import requests
import json
import logging

from stringutils import StringUtils

__author__ = "Rafael Gonçalves, Stanford University"


class TextAnnotator:

    def __init__(self, bp_api_key):
        self.url = "http://data.bioontology.org/annotator"
        self.bp_api_key = bp_api_key
        logging.basicConfig(level=logging.INFO)

    def annotate(self, text, annotations_limit=5):
        headers = {
            "Authorization": "apiKey token=" + self.bp_api_key,
        }
        params = {
            "text": text,
            "whole_word_only": "true",
            "longest_only": "true"
        }
        response = requests.get(self.url, params=params, headers=headers, verify=True)
        if response.ok:
            json_resp = json.loads(response.content)
            if len(json_resp) > 0:
                annotations = []
                for annotation in json_resp:
                    if len(annotations) < annotations_limit:
                        ann_class = annotation["annotatedClass"]
                        term_iri = ann_class["@id"]
                        term_link_bp = ann_class["links"]["self"]
                        onto_iri = ann_class["links"]["ontology"]
                        onto_name = onto_iri[onto_iri.rfind("/")+1:len(onto_iri)]
                        bp_link = ann_class["links"]["ui"]
                        match_type = annotation["annotations"][0]["matchType"]
                        matched_text = annotation["annotations"][0]["text"]
                        term_name, term_definition = self.get_term_name_and_def(term_link_bp)

                        a_details = Annotation(text, term_name, term_iri, term_definition, onto_iri, onto_name, bp_link,
                                               match_type, matched_text)
                        annotations.append(a_details)
                return annotations
            else:
                logging.info("Empty response for text input: " + text + ".")
                return []
        else:
            logging.error("Bad response: " + response.reason + ". Input: " + text + ".\n\tRequest URL: " +
                          response.url + "\n\tResponse: " + str(response))
            return []

    def get_term_name_and_def(self, term_iri):
        headers = {
            "Authorization": "apiKey token=" + self.bp_api_key,
        }
        response = requests.get(term_iri, headers=headers, verify=True)
        if response.ok:
            json_resp = json.loads(response.content)
            if len(json_resp) > 0:
                term_name = json_resp["prefLabel"]
                if term_name is not None:
                    term_name = self.remove_quotes(term_name)
                else:
                    term_name = "Unresolved"

                term_definition = ""
                if len(json_resp["definition"]) > 0:
                    term_definition = json_resp["definition"][0]
                    term_definition = self.remove_quotes(term_definition)
                return term_name, term_definition
            else:
                logging.info("Empty response for term search input: " + term_iri + ".")
                return '', ''
        elif response.status_code == 429:
            logging.info(response.reason + ".\n\tStatus code: " + str(response.status_code) + ". Waiting 15 seconds.")
            time.sleep(15)
            return self.get_term_name_and_def(term_iri)
        else:
            logging.error("Bad response: " + response.reason + ". Input: " + term_iri + ".\n\tRequest URL: " +
                          response.url + "\n\tStatus code: " + str(response.status_code))
            return '', ''

    def remove_quotes(self, text):
        text = text.replace("\"", "")
        text = text.replace("\'", "")
        return text

    def annotate_multiple(self, items, out_file):
        for item in items:
            annotations = self.annotate(item)
            StringUtils.save_list_to_file(out_file, annotations, mode='a')


class Annotation:

    def __init__(self, original_text, term_name, term_iri, term_definition, ontology_iri, ontology_name, bioportal_link,
                 match_type, matched_text):
        self.original_text = original_text
        self.term_name = term_name
        self.term_iri = term_iri
        self.term_definition = term_definition
        self.ontology_iri = ontology_iri
        self.ontology_name = ontology_name
        self.bioportal_link = bioportal_link
        self.match_type = match_type
        self.matched_text = matched_text

    def __str__(self):
        return self.original_text + "," + self.term_iri + ",\"" + self.term_name + "\",\"" + self.term_definition + \
               "\"," + self.ontology_name + "," + self.ontology_iri + "," + self.bioportal_link + "," + \
               self.match_type + "," + self.matched_text


if __name__ == "__main__":
    input_file = StringUtils.parse_file(sys.argv[1])
    output_file = sys.argv[2]
    ann = TextAnnotator(sys.argv[3])
    ann.annotate_multiple(input_file, output_file)
