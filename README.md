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

After opening the Notebook, execute the code cells in order.

## Project Structure
- `code/`: Contains the main code files of the project.
  - `torchmetric.py`: Defines metric calculation tools based on PyTorch.
  - `explainer.py`: Implements the model explainer.
  - `appprior.py`: Tools for calculating deep appearance priors.
  - `benchmark_app.ipynb`: The main entry point of the program, containing the project logic.
  - `mnist.ipynb`: Loads the MNIST dataset and integrates Deep-Prior-EG and other attribution priors into the training of the MNIST handwritten digit recognition model.
- `data/`: Contains the main image data for the project.
  - `imagenet50`: 50 similar-to-ImageNet images provided by the SHAP library, completely unused in model pretraining, suitable for validating the explanation methods in `benchmark_app.ipynb`.
  - `imagenet_prior`: Binary shape contour images of 12 ImageNet categories, manually annotated.
  - `ImageNet-1K`: ImageNet (ILSVRC) 2012, the most commonly used subset of ImageNet. The val folder under this directory is the validation set, which can be used for quantitative evaluation of explanation methods. It is recommended to apply for download from the official website: https://www.image-net.org/.
- `requirements.txt`: Project dependency file.

## Environment Requirements
- Python 3.9+
- CUDA support (optional, for accelerating deep learning models)
