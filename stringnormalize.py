#!/usr/bin/env python3
"""Provides StringNormalize class"""

import re

__author__ = "Rafael Gonçalves, Stanford University"


class StringNormalize:

    def __init__(self):
        self.pattern = re.compile('([^\s\w]|_)+')
        self.first_cap_re = re.compile('(.)([A-Z][a-z]+)')
        self.all_cap_re = re.compile('([a-z0-9])([A-Z])')

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
        token = self.parse_camel_case(token)
        token = re.sub('\s+', ' ', token)
        return self.to_ascii(token).strip()

    def parse_camel_case(self, name):
        s1 = self.first_cap_re.sub(r'\1_\2', name)
        s2 = self.all_cap_re.sub(r'\1_\2', s1).lower()
        pparts = re.split("[-_]", s2)
        name = " ".join([x.title() for x in pparts])
        return name.lower()

    def to_ascii(self, string):
        return string.encode("ascii", errors="ignore").decode()
