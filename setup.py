#!/usr/bin/env python
from setuptools import find_packages, setup

requirements = []
with open("requirements.txt") as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setup(
    name="huggingface_objectdetection",
    version="0.0.2",
    description="scivision plugin for Hugging Face object detection",
    author="Ed Chalstrey",
    author_email="echalstrey@turing.ac.uk",
    url="https://github.com/alan-turing-institute/scivision_huggingface_objectdetection",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
)
