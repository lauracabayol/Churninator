# Churninator

Churninator is a python module to make churn prediction with Gradient boosting and/or neural networks

## Installation
(Last update Aug 4 2024)

- Create a new conda environment. It is usually better to follow python version one or two behind, we recommend 3.11.

```
conda create -n churninator -c conda-forge python=3.11 pip=24.0
conda activate churninator
```

- Clone the repo into your machine and perform an *editable* installation:

```
git clone https://github.com/lauracabayol/Churninator.git 
cd Churninator
pip install -e .
``` 

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel` and `jupytext`:

```
pip install ipykernel
python -m ipykernel install --user --name churninator --display-name churninator
```


#### Tutorials:

In the `Notebooks` folder, there are several tutorials one can run to learn how to use the code.


