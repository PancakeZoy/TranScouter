from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()


# Read the version from transcouter/_version.py
def read_version():
    version_dict = {}
    with open(HERE / "transcouter" / "_version.py") as version_file:
        exec(version_file.read(), version_dict)
    return version_dict["__version__"]


setup(
    name="transcouter",
    version=read_version(),
    description="Transferable LLM-enhanced model for predicting genetic perturbation effect across datasets",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["transcouter", "transcouter.*"]),
    author="Ouyang Zhu, Jun Li",
    author_email="ozhu@nd.edu",
    url="https://github.com/PancakeZoy/transcouter",
    install_requires=[
        "torch >= 2.0.0",
        "tqdm >= 4.0.0",
        "anndata >= 0.10.0",
        "pandas >= 2.2.0",
        "numpy >= 1.21.0",
        "scanpy >= 1.10.0",
        "seaborn >= 0.13.0",
        "matplotlib >= 3.4.0",
        "scikit-learn >= 1.0.0",
        "scipy >= 1.5.0",
        "requests >= 2.0.0",
    ],
    python_requires=">=3.10.0",
)
