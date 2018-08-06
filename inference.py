import torch
from torch.autograd import Variable
import numpy as np
from FCN import FCNs
from FCN import VGGNet 



# the input to the model is of shape N*3*160*160 tensor, N presents the number of samples
x = np.random.rand(1, 3, 160, 160)
x = Variable(torch.FloatTensor(x))
model = torch.load('checkpoints/****.pt')
model = model.cpu()
# y is the output, of shape N*2*160*160, 2 present the class, [1 0] for background [0 1] for handbag
y = model(x)




