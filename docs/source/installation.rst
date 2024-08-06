Installation
============

To install Churninator, these are the recommended steps:

1. Create a new conda environment and activate it. It is usually better to follow a Python version one or two behind the latest. In January 2024, the latest is 3.12, so we recommend 3.11.

   .. code-block:: bash

    conda create -n churninator -c conda-forge python=3.11 pip=24.0
    conda activate churninator

2. Clone the repo into your machine and perform an editable installation:

   .. code-block:: bash

    git clone https://github.com/lauracabayol/Churninator.git 
    cd Churninator
    pip install -e .


   If you want to use notebooks via JupyterHub, you'll also need to download ipykernel:

   .. code-block:: bash

      pip install ipykernel
      python -m ipykernel install --user --name churninator --display-name churninator

   Furthermore, the LaCE repo uses jupytext to handle notebooks and version control. To run the notebooks provided in the repo:

   .. code-block:: bash

      pip install jupytext

   .. code-block:: bash

      jupytext your_script.py --to ipynb

3. Since the data used to develop the code is not my own, it is not publicly stored in the repo. Please save the data in the `data` direcotry and run the preprocessing script. This is in the `scripts` directory. This will create a new csv data file with the preprocessed data. It might take ~5', butit only requires running once.
   .. code-block:: bash

    python preprocessing_data.py \
        --path_to_data '../../data/Assignment_churn_prediction.xlsx' \
        --labels_to_encode Gender Country \
        --save_file_path '../../data/clean_data.csv' \
        --verbose True 


