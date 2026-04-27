from setuptools import setup, find_packages

setup(
    name="physiminer",
    version="1.0.0",
    description="Python port of PhysiMineR: extraction and analysis of oesophageal physiology data",
    author="Sebastian Zeki (Python port)",
    license="GPL-3",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.23",
    ],
)
