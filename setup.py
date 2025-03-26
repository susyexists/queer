from setuptools import setup, find_packages

setup(
    name="queer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "joblib",
        "psutil",
        "tqdm"
    ],
    author="QUEER Development Team",
    author_email="susy@materials.wiki",
    description="Quantum Utilities and Electron Engineering Resources",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/susyexists/queer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
)