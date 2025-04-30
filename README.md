Here's the translated content in Markdown format:

DeepPrior-EG

Installation Dependencies
First install the required dependencies:
```bash
pip install -r requirements.txt
```

Running the Project
Execute the following command to run the Jupyter Notebook:
```bash
jupyter notebook ./code/benchmark_app.ipynb
```

After opening the Notebook, run code cells sequentially.

Project Structure
• `code/`: Contains main code files

  • `torchmetric.py`: PyTorch-based metric calculation utilities

  • `explainer.py`: Model explainer implementations

  • `appprior.py`: Deep appearance prior probability calculator

  • `benchmark_app.ipynb`: Main entry point with project execution logic

• `data/`: Contains image datasets

  • `imagenet50`: 50 SHAP-provided ImageNet-style images (never used in model pre-training) for validating explanation methods

  • `imagenet_prior`: Manually annotated binary contour images for 12 ImageNet categories

  • `ImageNet-1K`: ImageNet (ILSVRC) 2012 subset

    ◦ `val`: Validation set for quantitative evaluation of explanation methods

• `requirements.txt`: Dependency list


Environment Requirements
• Python 3.9+

• CUDA support (optional for DL acceleration)
