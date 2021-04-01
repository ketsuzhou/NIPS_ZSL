import torch
from VAE import Attr_AutoEncoder
import numpy as np

file1 = open('../../../media/data/AwA2/predicate-matrix-binary.txt')
# file2 = open('../../../media/data/AwA2/predicate-matrix-continuous.txt')
# Attr_AutoEncoder

attributes = torch.zeros(50, 85)
for i in range(50):
    a = file1.readline()
    dlist = a.split()
    attributes[i][:] = torch.tensor([int(x) for x in dlist])

attributes = attributes.cuda()
model = Attr_AutoEncoder(85, 256, 512).cuda()
beta = 3
for epoch in range(100):
    recon_loss, kl_loss = model(attributes)
    loss = recon_loss + beta * kl_loss

torch.save(model, 'net.pkl')
