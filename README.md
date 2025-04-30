DeepPrior-EG

Installation Dependencies
First install the required dependencies for the project:
```bash
pip install -r requirements.txt
```

Running the Project
Execute the following command to run the Jupyter Notebook in the project:
```bash
jupyter notebook ./code/benchmark_app.ipynb
```

After opening the Notebook, run the code cells sequentially.

Project Structure
• `code/`: Contains the main code files of the project.

  • `torchmetric.py`: Defines PyTorch-based metric calculation utilities.

  • `explainer.py`: Provides implementations of model explainers.

  • `appprior.py`: Tool for computing deep appearance prior probabilities.

  • `benchmark_app.ipynb`: Main program entry point containing the project's execution logic.

• `data/`: Contains the primary image data for the project.

  • `imagenet50`: 50 ImageNet-like images provided by the SHAP library, completely excluded from model pre-training. These can be used to validate explanation methods in `benchmark_app.ipynb`.

  • `imagenet_prior`: Binary contour images of 12 ImageNet categories, manually annotated.

  • `ImageNet-1K`: ImageNet (ILSVRC) 2012, the most commonly used subset of ImageNet. The `val` subfolder contains the validation set for quantitative evaluation of explanation methods.

• `requirements.txt`: Project dependency file.


Environment Requirements
• Python 3.9+

• CUDA support (optional, for accelerating deep learning models)
