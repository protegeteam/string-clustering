#!/usr/bin/env python3
"""Provides StringUtils class"""

import json
import pandas as pd

__author__ = "Rafael Gonçalves, Stanford University"


class StringUtils:

    @staticmethod
    def get_dictionary_as_json(dictionary):
        return json.dumps(dictionary, sort_keys=True, indent=2)

    @staticmethod
    def parse_file(input_file):
        file = open(input_file)
        lines = file.read().splitlines()
        return lines

    @staticmethod
    def save_dictionary_as_json(output_file, output):
        output_file = open(output_file, "w+")
        output_file.write(StringUtils.get_dictionary_as_json(output))
        output_file.close()

    @staticmethod
    def save_distances_matrix(output_file, distances, tokens):
        names = [t for t in tokens]
        df = pd.DataFrame(distances, index=names, columns=names)
        df.to_csv(output_file, index=True, header=True, sep=',')
