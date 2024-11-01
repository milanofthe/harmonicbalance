from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="harmonicbalance",
    version="0.2.0",
    author="Milan Rother",
    author_email="milan.rother@gmx.de",
    description="Minimalistic framework for calculating nonlinear periodic steady state responses using an object based harmonic balance.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/milanofthe/harmonicbalance",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
    ],
)