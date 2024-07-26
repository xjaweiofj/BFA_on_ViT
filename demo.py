import torch
import numpy
from utils import AverageMeter

x = torch.randn(3, 5)
print (x)

x = x.view(-1)
print(x.size())
print (x)



# Initialize a meter to record loss
losses = AverageMeter()
print (losses.avg)
x = torch.randn(3, 5)
loss_value = x
print (x)
batch_size = 5
# Update meter after every minibatch update
losses.update(loss_value, batch_size)

print (losses.val)
print (losses.avg)
