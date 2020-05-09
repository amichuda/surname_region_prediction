# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

with open("README.md") as f:
    LONG_DESCRIPTION, LONG_DESC_TYPE = f.read(), "text/markdown"

NAME = "uganda_surname_predict"
AUTHOR_NAME, AUTHOR_EMAIL = "Aleksandr Michuda", "amichuda@ucdavis.edu"
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Intended Audience :: Social Scientists/Researchers"
    "Topic :: Scientific/Economics/Nutrition",
]
LICENSE = "MIT"
DESCRIPTION = "Predictor of Ugandan region of origin based on surname"
URL = "https://github.com/amichuda/uber_surname_region_prediction"
PYTHON_REQ = ">=3.5"

setup(
    name=NAME,
    version='0.1dev',
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    python_requires=PYTHON_REQ,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    classifiers=CLASSIFIERS,
    install_requires=requirements.txt,
    packages=find_packages(),
)
