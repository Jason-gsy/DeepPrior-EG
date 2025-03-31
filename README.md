# DeepPrior-EG

## Install Dependencies
First, install the required dependencies for the project:
```bash
pip install -r requirements.txt
```

## Run the Project
Execute the following command to run the Jupyter Notebook in the project:
```bash
jupyter notebook ./code/benchmark_app.ipynb
```

Open the Notebook and execute the cells in order.

## Project Structure
- `code/`: Contains the main code files of the project.
  - `torchmetric.py`: Defines PyTorch-based metric calculation tools.
  - `explainer.py`: Provides the implementation of the model explainer.
  - `appprior.py`: Computes deep appearance prior probabilities.
  - `benchmark_app.ipynb`: The main entry point of the project, containing the execution logic.
- `data/`: Contains the main image data for the project.
- `requirements.txt`: Dependency file for the project.

## Environment Requirements
- Python 3.9+
- CUDA support (optional, for accelerating deep learning models)
