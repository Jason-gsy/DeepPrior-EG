# DeepPrior-EG

## 安装依赖
首先安装项目所需的依赖：
```bash
pip install -r requirements.txt
```

## 运行项目
执行以下命令运行项目中的 Jupyter Notebook：
```bash
jupyter notebook ./code/benchmark_app.ipynb
```

打开 Notebook 后，按照代码单元顺序运行即可。

## 项目结构
- `code/`: 包含项目的主要代码文件。
  - `torchmetric.py`: 定义了基于 PyTorch 的指标计算工具。
  - `explainer.py`: 提供了模型解释器的实现。
  - `appprior.py`: 计算深度外观先验概率的工具。
  - `benchmark_app.ipynb`: 主程序入口，包含项目的运行逻辑。
- `data/`: 包含项目的主要图片数据。
  - `imagenet50`: SHAP库中提供的50幅类似imgagenet的图像，完全未参与模型预训练的图片，可用于验证benchmark_app.ipynb中的解释方法。
  - `imagenet_prior`: 12个类别的imagenet图像的形状轮廓二值图像，由人工标注。
  - `ImageNet-1K`: ImageNet (ILSVRC) 2012，这是 ImageNet 最常用的子集。该文件夹下val是验证集，可用于解释方法的定量评估。
- `requirements.txt`: 项目依赖文件。

## 环境要求
- Python 3.9+
- CUDA 支持（可选，用于加速深度学习模型）
