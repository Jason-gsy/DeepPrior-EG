import experiments as experiments

import torch
import numpy as np
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = torch.load('/u01/guosuying/XAI-prior/ExpectedGradients_IntegratedGradients_pytorch/model/cnn.pt').to('cuda')
# idx = 43
# target = 7
# n_iter = 1000
# trainset, testset = load_dataset()
# baseline = testset.test_data.view(len(testset), 1, 28,28).float().numpy()
# data = torch.from_numpy(baseline[np.where(testset.test_labels == target)][idx:idx+1]).to('cuda')

print(experiments.run_experiments(dataset='independentlinear60', model='ffnn', method='expected_gradients', metric='keep_positive_mask', cache_dir="/tmp", nworkers=1))