import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import functools

class ActNorm(nn.Module):
    def __init__(self, num_features, affine=True, logdet=False):
        super().__init__()
        assert affine
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        return output / self.scale - self.loc        

# _norm_options = {
#         "in": nn.InstanceNorm2d,
#         "bn": nn.BatchNorm2d,
#         "an": ActNorm}


class Distribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 10.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5*self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def sample(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = self.mean + self.std*torch.randn(self.mean.shape).to(device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5*torch.sum(torch.pow(self.mean, 2)
                        + self.var - 1.0 - self.logvar,
                        dim=[1,2,3])
            else:
                return 0.5*torch.sum(
                        torch.pow(self.mean - other.mean, 2) / other.var
                        + self.var / other.var - 1.0 - self.logvar + other.logvar,
                        dim=[1,2,3])

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0*np.pi)
        return 0.5*torch.sum(
                logtwopi+self.logvar+torch.pow(sample-self.mean, 2) / self.var,
                dim=[1,2,3])

class DenseEncoderLayer(nn.Module):
    def __init__(self, scale, kernel_size, out_channels, in_channels=None):
        """[summary]
        Args:
            scale ([type]): [description]
            spatial_size ([type]): [description]
            out_channels ([type]): [description]
            in_channels ([type], optional): [description]. Defaults to None.
        """
        super().__init__()
        self.scale = scale
        self.in_channels = 64*min(2**(self.scale-1), 16)
        if in_channels is not None:
            self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sub_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                bias=True)])

    def forward(self, input):
        for sub_layer in self.sub_layers:
            input = sub_layer(input)
        return input


class DenseDecoderLayer(nn.Module):
    def __init__(self, scale, kernel_size, in_channels):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = 64*min(2**self.scale, 16)
        self.kernel_size = kernel_size
        self.sub_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                bias=True)])

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x


class FeatureLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm='nn.InstanceNorm2d'):
        super().__init__()
        self.scale = scale
        self.norm = eval(norm) 
        if in_channels is None:
            self.in_channels = 64*min(2**(self.scale-1), 16)
        else:
            self.in_channels = in_channels
        Norm = functools.partial(self.norm, affine=True)
        Activate = lambda: nn.LeakyReLU(0.2)
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=64*min(2**self.scale, 16),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                Norm(num_features=64*min(2**self.scale, 16)),
                Activate()])

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm='nn.InstanceNorm2d'):
        super().__init__()
        self.scale = scale
        self.norm = norm
        if in_channels is not None:
            self.in_channels = in_channels
        else:
            self.in_channels = 64*min(2**(self.scale+1), 16)
        Norm = functools.partial(self.norm, affine=True)
        Activate = lambda: nn.LeakyReLU(0.2)
        self.sub_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=64*min(2**self.scale, 16),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                Norm(num_features=64*min(2**self.scale, 16)),
                Activate()])

    def forward(self, input):
        d = input
        for layer in self.sub_layers:
            d = layer(d)
        return d

class ImageLayer(nn.Module):
    def __init__(self, out_channels=3, in_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        FinalActivate = lambda: torch.nn.Tanh()
        self.sub_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                FinalActivate()
                ])    

    def forward(self, input):
        x = input
        for sub_layer in self.sub_layers:
            x = sub_layer(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Model(nn.Module):  
    def __init__(self, n_down,z_dim,in_size,in_channels,norm,deterministic):
        """[summary]
        Args:
            n_down ([type]): [description]
            z_dim ([type]): [the dimension of latent variable z]
            in_size ([type]): [input size]
            in_channels ([type]): [input channels]
            norm ([norm_options = {"in": nn.InstanceNorm2d, "bn": nn.BatchNorm2d, "an": ActNorm}]): [specifying normalization method to input of each layer]
            deterministic ([bool]): [description]
        """
        super().__init__()
        # use cudnn to speed up in case of the size of model not changing during training 
        cudnn.benchmark = True
        n_down = n_down
        z_dim = z_dim
        in_size = in_size
        bottleneck_size = in_size // 2**n_down
        in_channels = in_channels
        norm = norm
        self.be_deterministic = deterministic

        self.feature_layers = nn.ModuleList()

        self.feature_layers.append(FeatureLayer(0, in_channels=in_channels, norm=norm))
        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale, norm=norm))

        self.dense_encode = DenseEncoderLayer(n_down, 1, 2*z_dim)
        self.dense_decode = DenseDecoderLayer(n_down-1, bottleneck_size, z_dim)
        
        self.decoder_layers = nn.ModuleList()
        for scale in range(n_down-1):
            self.decoder_layers.append(DecoderLayer(scale, norm=norm))
        self.image_layer = ImageLayer(out_channels=in_channels)

        self.apply(weights_init)

        self.n_down = n_down
        self.z_dim = z_dim
        self.bottleneck_size = bottleneck_size

    def encode(self, input):
        h = input
        for feature_layer in self.feature_layers:
            h = feature_layer(h)
        h = self.dense_encode(h)
        return Distribution(h, deterministic=self.be_deterministic)

    def decode(self, input):
        h = input
        h = self.dense_decode(h)
        for decoder_layer in reversed(self.decoder_layers):
            h = decoder_layer(h)
        h = self.image_layer(h)
        return h

    def get_last_layer(self):
        """
        Returns:
            [type]: [get the weight of last layer]
        """        
        return self.image_layer.sub_layers[0].weight




if __name__ == "__main__":
    batch_size= 50
    # pre_calc_stat_path="./fid_stats/cmnist.npz"
    device = torch.device("cuda")
    model=Model(n_down=4,z_dim=64,in_size=32,in_channels=3,norm='ActNorm',deterministic=False).cuda()
