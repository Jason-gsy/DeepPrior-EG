import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import cv2
import os
import numpy as np
from scipy.ndimage import zoom
from sklearn.mixture import BayesianGaussianMixture
# 加载预训练的ResNet50模型，并提取特定层的输出作为特征向量
def load_resnet50_feature_extractor():
    resnet50 = models.resnet50(pretrained=True)
    # 移除最后两层（全局平均池化层和全连接层）

    # 将模型截断到中间层
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-2])
    return feature_extractor
# 加载预训练的vgg16模型，并提取特定层的输出作为特征向量
def load_vgg16_feature_extractor():
    vgg16 = models.vgg16(pretrained=True)
    # 保留所有卷积层
    feature_extractor = vgg16.features
    return feature_extractor
# 加载预训练的resnet152模型，并提取特定层的输出作为特征向量
def load_resnet152_feature_extractor():
    resnet152 = models.resnet152(pretrained=True)

    # 将模型截断到中间层
    feature_extractor = torch.nn.Sequential(*list(resnet152.children())[:-2])
    return feature_extractor


# 加载并预处理图像
def load_and_preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = preprocess(image).unsqueeze(0)  # 添加一个批次维度
    return image

def load_and_preprocess_image_one(image):
    # 将图像大小调整为256x256
    factors = [256 / dim for dim in image.shape[:2]]
    image = zoom(image, factors + [1], order=1)  # [1]是为了保持通道数不变

    # 对图像进行中心裁剪，得到224x224的图像
    start = [(dim - 224) // 2 for dim in image.shape[:2]]
    image = image[start[0]:start[0]+224, start[1]:start[1]+224, :]

    # 将图像从0-255转换为0-1
    image = image / 255.0

    # 对图像进行归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # 将图像从HWC转换为CHW
    image = image.transpose((2, 0, 1))

    # 添加一个批次维度
    image = image[np.newaxis, ...]

    return image
# 计算特征向量
def extract_features(image_path, feature_extractor):
    if type(image_path) == np.ndarray:
        image = load_and_preprocess_image_one(image_path)
        image = torch.from_numpy(image)
        image = image.float()
    else:
        image = load_and_preprocess_image(image_path)

    with torch.no_grad():
        features = feature_extractor(image)
    return features.squeeze().numpy()  # 移除批次维度并转换为NumPy数组

# 计算外观先验概率图
def compute_appearance_prior(features):
    features_of_interest = features.reshape(-1, features.shape[-1])
    print("features_of_interest:",features_of_interest.shape)
    # 计算特征向量的样本均值和协方差矩阵
    sample_mean = np.mean(features_of_interest, axis=0)
    sample_covariance = np.cov(features_of_interest, rowvar=False)
    #输出sample_mean中的数据类型，是float还是double
    print("sample_mean:",sample_mean.dtype)
    # # 添加小的正则化项
    # epsilon = 1e-6
    # sample_covariance += np.eye(sample_covariance.shape[0]) * epsilon

    # 构建多元高斯分布
    multivariate_normal = torch.distributions.MultivariateNormal(torch.tensor(sample_mean), torch.tensor(sample_covariance))

    # 对于每个像素点，计算其属于指定类别的概率
    appearance_prior = multivariate_normal.log_prob(torch.tensor(features))
    probability_map = torch.exp(appearance_prior)
    print("probability_map:",probability_map.shape)
    # 将概率图调整为目标大小（224x224）
    resized_probability_map = F.interpolate(probability_map.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

    # 移除批次和通道维度
    resized_probability_map = resized_probability_map.squeeze().squeeze()
    

    return resized_probability_map
#通过贝叶斯高斯混合模型计算外观先验概率图
def compute_appearance_postprior(features):
    features_of_interest = features.reshape(-1, features.shape[-1])
    print("features_of_interest:",features_of_interest.shape)
    bgmm = BayesianGaussianMixture(n_components=5, covariance_type='full', max_iter=100)
    bgmm.fit(features_of_interest)

    # Compute posterior probabilities with reshaped features
    posterior_probs = bgmm.predict_proba(features_of_interest)
    posterior_probs = torch.tensor(posterior_probs)
    resized_probability_map = F.interpolate(posterior_probs.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    resized_probability_map = resized_probability_map.squeeze().squeeze()


    return resized_probability_map
#计算一个文件夹下所有图像的概率图，并加和平均作为先验概率分布
def compute_appearance_prior_all(path,model):
    files = os.listdir(path)
    #读取以jpeg结尾的文件
    files = [file for file in files if file.endswith(".JPEG")]
    #选择前200张图像
    files = files[:200]
    appearance_prior = None
    for file in files:
        image_path = os.path.join(path, file)
        if model == "resnet50":
            # 加载ResNet50特征提取器
            feature_extractor = load_resnet50_feature_extractor()
        elif model == "resnet152":
            # 加载ResNet152特征提取器
            feature_extractor = load_resnet152_feature_extractor()
        else:
            # 加载VGG16特征提取器
            feature_extractor = load_vgg16_feature_extractor()
        # 提取特征向量
        features = extract_features(image_path, feature_extractor)
        # 计算外观先验概率图
        appearance_prior = compute_appearance_prior(features)
        # appearance_prior = compute_appearance_postprior(features)
        if appearance_prior is None:
            appearance_prior = appearance_prior
        else:
            appearance_prior = appearance_prior + appearance_prior
    appearance_prior = appearance_prior / len(files)
    return appearance_prior
#计算一个图像的概率图，并加和平均作为先验概率分布
def compute_appearance_prior_one(image,model,post=False):
    if model == "resnet50":
        # 加载ResNet50特征提取器
        feature_extractor = load_resnet50_feature_extractor()
    elif model == "resnet152":
        # 加载ResNet152特征提取器
        feature_extractor = load_resnet152_feature_extractor()
    else:
        # 加载VGG16特征提取器
        feature_extractor = load_vgg16_feature_extractor()
    # 提取特征向量
    features = extract_features(image, feature_extractor)
    # 计算外观先验概率图
    if post:
        print("postprior")
        appearance_prior = compute_appearance_postprior(features)
        appearance_prior = np.array(appearance_prior)
    else:
        appearance_prior = compute_appearance_prior(features)
        appearance_prior = np.array(appearance_prior)*10
    return appearance_prior


if __name__ == '__main__':
    image_path = "/u01/guosuying/XAI-prior/shap_bench/data/imagenet50/sim_n01549053_4208.jpg"  # 替换为您的图像路径
    
    # 加载ResNet50特征提取器
    feature_extractor = load_resnet50_feature_extractor()
    # 加载VGG16特征提取器
    # feature_extractor = load_vgg16_feature_extractor()
    
    # 提取特征向量
    features = extract_features(image_path, feature_extractor)
    
    # 计算外观先验概率图
    appearance_prior = compute_appearance_prior(features)
    postprior = compute_appearance_postprior(features)
    # print("外观先验概率图:", postprior)
    # #输出概率图最大值和最小值
    # print("最大值:", postprior.max())
    # print("最小值:", postprior.min())
    print("大小:", appearance_prior.shape)
    # print("大小:", postprior.shape)

