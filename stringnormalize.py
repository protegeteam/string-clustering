#!/usr/bin/env python3
"""Provides StringNormalize class"""

import re

__author__ = "Rafael Gonçalves, Stanford University"


class StringNormalize:

    def __init__(self):
        self.pattern = re.compile('([^\s\w]|_)+')

    def normalize_tokens(self, tokens):
        normalized_tokens = set()
        for token in tokens:
            normalized_token = self.normalize(token)
            if normalized_token != "":
                normalized_tokens.add(normalized_token)
        return normalized_tokens

    # replace all non-alphanumeric characters with spaces, and trim all extra white space
    def normalize(self, token):
        token = self.pattern.sub(' ', token)
        return re.sub('\s+', ' ', token).strip()
