#!/usr/bin/env python3
"""Provides TextAnnotator class"""

import sys
import time
import requests
import json
import logging

from stringutils import StringUtils
from enum import Enum

__author__ = "Rafael Gonçalves, Stanford University"


class TextAnnotator:

    def __init__(self, bp_api_key):
        self.url = "http://data.bioontology.org/annotator"
        self.bp_api_key = bp_api_key
        logging.basicConfig(level=logging.INFO)

    def annotate(self, text, ontologies, annotations_limit=5):
        params = {
            "text": text,
            "whole_word_only": "true",
            "longest_only": "true",
            "ontologies": ontologies
        }
        annotations = []
        logging.info("Searching for ontology terms to match: " + text)
        response = self.do_get_request(self.url, params=params)
        if response is not None:
            logging.info("\tFound " + str(len(response)) + " annotation(s)")
            for annotation in response:
                if len(annotations) < annotations_limit:
                    annotations.append(self.get_annotation_details(text, annotation))
        return annotations

    def get_annotation_details(self, text, annotation):
        ann_class = annotation["annotatedClass"]
        term_iri = ann_class["@id"]
        term_link_bp = ann_class["links"]["self"]
        onto_iri = ann_class["links"]["ontology"]
        onto_name = onto_iri[onto_iri.rfind("/") + 1:len(onto_iri)]
        bp_link = ann_class["links"]["ui"]
        match_type = annotation["annotations"][0]["matchType"]
        matched_text = annotation["annotations"][0]["text"]
        term_name, term_definition, ancestors = self.get_term_details(term_link_bp)
        return Annotation(text, term_name, term_iri, term_definition, ancestors, onto_iri, onto_name, bp_link,
                          match_type, matched_text)

    def get_term_details(self, term_iri):
        response = self.do_get_request(term_iri)
        term_name, term_definition = "", ""
        ancestors = []
        if response is not None:
            term_name = StringUtils.remove_quotes(response["prefLabel"])
            if len(response["definition"]) > 0:
                term_definition = response["definition"][0]
                term_definition = StringUtils.remove_quotes(term_definition)
            ancestors_link = response["links"]["ancestors"]
            ancestors = self.get_ancestors(ancestors_link)
        return term_name, term_definition, ancestors

    def get_ancestors(self, term_ancestors_bp_link):
        response = self.do_get_request(term_ancestors_bp_link)
        ancestors = []
        for ancestor in response:
            if ancestor is not None:
                ancestor_name = ancestor["prefLabel"]
                ancestors.append(ancestor_name)
        ancestors = list(dict.fromkeys(ancestors))  # remove duplicate ancestors
        return ancestors

    def do_get_request(self, request_url, params=None):
        headers = {
            "Authorization": "apiKey token=" + self.bp_api_key,
        }
        response = requests.get(request_url, params=params, headers=headers, verify=True)
        if response.ok:
            json_resp = json.loads(response.content)
            if len(json_resp) > 0:
                return json_resp
            else:
                logging.error("Empty response for input: " + request_url + " with parameters " + str(params))
        elif response.status_code == 429:
            logging.info(response.reason + ".\n\tStatus code: " + str(response.status_code) + ". Waiting 15 seconds.")
            time.sleep(15)
            return self.do_get_request(request_url, params)
        else:
            json_resp = json.loads(response.content)
            logging.error(response.reason + ":" + request_url + ".\t" + json_resp["errors"][0])

    def annotate_list_and_append_to_file(self, items, out_file, ontologies):
        csv_header = "original_text,term_iri,term_name,term_definition,ancestors,ontology_name,ontology_iri," \
                     "bioportal_link,match_type,matched_text"
        StringUtils.save_line_to_file(out_file, csv_header, mode='a')
        logging.info("Matching against ontologies: " + ontologies)
        for item in items:
            annotations = self.annotate(item, ontologies)
            StringUtils.save_list_to_file(out_file, annotations, mode='a')


class Annotation:

    def __init__(self, original_text, term_name, term_iri, term_definition, term_ancestors, ontology_iri, ontology_name,
                 bioportal_link, match_type, matched_text):
        self.original_text = original_text
        self.term_name = term_name
        self.term_iri = term_iri
        self.term_definition = term_definition
        self.term_ancestors = term_ancestors
        self.ontology_iri = ontology_iri
        self.ontology_name = ontology_name
        self.bioportal_link = bioportal_link
        self.match_type = match_type
        self.matched_text = matched_text

    def __str__(self):
        ancestors_str = StringUtils.remove_brackets(str(self.term_ancestors))
        ancestors_str = StringUtils.remove_quotes(ancestors_str)
        return self.original_text + "," + self.term_iri + ",\"" + self.term_name + "\",\"" + self.term_definition + \
            "\",\"" + ancestors_str + "\"," + self.ontology_name + "," + self.ontology_iri + "," + \
            self.bioportal_link + "," + self.match_type + "," + self.matched_text


# Enumeration of commonly-used ontologies and their BioPortal acronyms
class Ontology(Enum):
    ALL = ""
    UBERON = "UBERON"
    CELL = "CL"
    MESH = "MESH"
    SNOMED = "SNOMEDCT"
    FMA = "FMA"
    NCIT = "NCIT"


if __name__ == "__main__":
    input_file = StringUtils.parse_file(sys.argv[1])  # a list of terms, one term per line
    output_file = sys.argv[2]  # the output file where the details of ontology term mappings
    bioportal_apikey = sys.argv[3]  # BioPortal API key

    onto = Ontology.ALL.value
    if len(sys.argv) > 4:
        onto = sys.argv[4]  # comma-separated list of ontologies

    annotator = TextAnnotator(bioportal_apikey)
    annotator.annotate_list_and_append_to_file(input_file, output_file, onto)
