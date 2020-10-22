#!/usr/bin/env python3
"""Provides StringUtils class"""

import json
import re

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
    def parse_cluster_dict(cluster_dict):
        with open(cluster_dict) as f:
            data = json.load(f)
        return data

    @staticmethod
    def save_list_to_file(output_file, str_list, mode='w'):
        with open(output_file, mode) as f:
            for item in str_list:
                f.write("%s\n" % item)

    @staticmethod
    def save_line_to_file(output_file, line, mode='w'):
        with open(output_file, mode) as f:
            f.write(line + '\n')

    @staticmethod
    def save_dictionary_as_json(output_file, output):
        output_file = open(output_file, "w+")
        output_file.write(StringUtils.get_dictionary_as_json(output))
        output_file.close()

    @staticmethod
    def tokenize_multi_word_strings(tokens):
        normalized_tokens = []
        for token in tokens:
            normalized_tokens.append(token.split())
        return normalized_tokens

    @staticmethod
    def remove_quotes(text):
        text = text.replace("\"", "")
        text = text.replace("\'", "")
        return text

    @staticmethod
    def remove_brackets(text):
        text = text.replace("[", "")
        text = text.replace("]", "")
        return text

    @staticmethod
    def remove_non_alphanumeric_chars(text):
        regex = re.compile('[^a-zA-Z0-9, ]')
        return regex.sub('', text)
