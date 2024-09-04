'''
This code is based on https://github.com/bhpfelix/Variational-Autoencoder-PyTorch
and has modifications proposed in https://arxiv.org/abs/2102.05692 like loss function and learning rate.
Finally we changed the layer dimensions to accept images with three color channels.
'''

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class AutoEncoder(nn.Module):
  def __init__(self, channels, init_output_size, latent_variable_size):
    super(AutoEncoder, self).__init__()
    self.channels = channels
    self.init_output_size = init_output_size
    self.latent_variable_size = latent_variable_size

    #ENCODER
    self.conv1 = nn.Conv2d(3, init_output_size, kernel_size=4, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(init_output_size)

    self.conv2 = nn.Conv2d(init_output_size, init_output_size*2, kernel_size=4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(init_output_size*2)

    self.conv3 = nn.Conv2d(init_output_size*2, init_output_size*4, kernel_size=4, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(init_output_size*4)

    self.conv4 = nn.Conv2d(init_output_size*4, init_output_size*8, kernel_size=4, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(init_output_size*8)

    self.conv5 = nn.Conv2d(init_output_size*8, init_output_size*8, kernel_size=4, stride=2, padding=1)
    self.bn5 = nn.BatchNorm2d(init_output_size*8)

    self.lin1 = nn.Linear(init_output_size*8*10*5, latent_variable_size)


    #DECODER
    self.lin2 = nn.Linear(latent_variable_size, init_output_size*8*10*5)
    self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd1 = nn.ReplicationPad2d(1)
    self.d2 = nn.Conv2d(1024, 1024, 3, 1)
    self.bn6 = nn.BatchNorm2d(1024, 1.e-3)

    self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd2 = nn.ReplicationPad2d(1)
    self.d3 = nn.Conv2d(1024, 512, 3, 1)
    self.bn7 = nn.BatchNorm2d(512, 1.e-3)

    self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd3 = nn.ReplicationPad2d(1)
    self.d4 = nn.Conv2d(512, 256, 3, 1)
    self.bn8 = nn.BatchNorm2d(256, 1.e-3)

    self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd4 = nn.ReplicationPad2d(1)
    self.d5 = nn.Conv2d(256, 128, 3, 1)
    self.bn9 = nn.BatchNorm2d(128, 1.e-3)

    self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
    self.pd5 = nn.ReplicationPad2d(1)
    self.d6 = nn.Conv2d(128, 3, 3, 1)

    self.leakyrelu = nn.LeakyReLU(0.2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def encode(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.leakyrelu(x)
    l1 = x

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.leakyrelu(x)
    l2 = x

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.leakyrelu(x)
    l3 = x

    x = self.conv4(x)
    x = self.bn4(x)
    x = self.leakyrelu(x)
    l4 = x

    x = self.conv5(x)
    x = self.bn5(x)
    x = self.leakyrelu(x)
    l5 = x

    x = x.reshape(-1,51200)
    x = self.lin1(x)

    return x,l1,l2,l3,l4,l5

  def decode(self, z):
    h1 = self.relu(self.lin2(z))
    h1 = h1.view(-1,1024,10,5)
    dec1 = h1
    h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
    dec2 = h2
    h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
    dec3 = h3
    h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
    dec4 = h4
    h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))
    dec5 = h5

    return self.sigmoid(self.d6(self.pd5(self.up5(h5)))), dec1, dec2, dec3, dec4, dec5

  def get_latent_var(self, x):
      z,l1,l2,l3,l4,l5 = self.encode(x) 
      return z

  def loss_layer(self,e1,e2,e3,e4,e5,d1,d2,d3,d4,d5,alfa=0.01):

    encode_layers = [e1,e2,e3,e4,e5]
    decode_layers = [d5,d4,d3,d2,d1]
    loss=0
    criterion = nn.MSELoss()

    for i in range(5): #number of encoded and decode layers
      loss+= criterion(encode_layers[i],decode_layers[i])

    loss = loss*alfa

    return loss

  def forward(self, x):
    z,e1,e2,e3,e4,e5 = self.encode(x)
    res, d1,d2,d3,d4,d5= self.decode(z)
    loss = self.loss_layer(e1,e2,e3,e4,e5,d1,d2,d3,d4,d5)
    return res,loss

def loss_function(recon_x, x):

  MSE = reconstruction_function(recon_x, x)

  return MSE

reconstruction_function = nn.MSELoss()

gpu = torch.device("cuda")
model = AutoEncoder(channels=3, init_output_size=128, latent_variable_size=1000).to(gpu)
optimizer = optim.Adam(model.parameters(), lr=1e-4)