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
#### Data:
```
Since the data used to develop the code is not my own, it is not publicly stored in the repo. Please save the data in the `data` direcotry and run
the preprocessing script. This is in the `scripts` directory.
```
```Example:
`python run_data_cleaner.py \
    --path_to_data '../data/Assignment_churn_prediction.xlsx' \
    --labels_to_encode Gender Country \
    --save_file_path '../data/clean_data.csv' \
    --verbose True \
    --make_plots False`


#### Tutorials:

In the `notebooks` folder, there are two tutorials, one for the Gradient boosting and the other for the neural network.
These are .py scripts, in order to pair them to .ipynb, please run:

```
jupytext your_script --to ipynb
```


