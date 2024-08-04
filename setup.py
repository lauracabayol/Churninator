import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "churninator",
    version = "1.0.0",
    author = "Laura Cabayol-Garcia",
    author_email = "lauracabayol@gmail.com",
    description = ("Gradient boosting and neural network for churn prediction"),
    keywords = "astronomy",
    url = "https://github.com/lauracabayol/Churninator.git",
    license="GPLv3",
    packages=['Churninator'],
    install_requires=['scikit-learn',
                      'torch',
                      'numpy', 
		      'pandas',
                      'matplotlib',
                      'torch',
                      'imblearn',
                      'seaborn',
                      'transformers',
                      'optuna',
                      'ipykernel',
                      'jupytext'],

    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)
