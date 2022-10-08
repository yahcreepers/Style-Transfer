import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        features = models.vgg19().features
        self.block1 = features[:2]    #VGG19 relu1_1
        self.block2 = features[2:7]   #VGG19 relu2_1
        self.block3 = features[7:12]  #VGG19 relu3_1
        self.block4 = features[12:21] #VGG19 relu4_1
#        for i in range(1, 5):
#            for param in getattr(self, f'block{i}').parameters():
#                param.requires_grad = False
    def forward(self, image):
#        test = models.vgg19(pretrained=True).cuda(0).features[:21]
#        h = test(image)
#        print(h)
        #print(image.shape)
        h_1 = self.block1(image)
        #print(h_1.shape)
        h_2 = self.block2(h_1)
        #print(h_2.shape)
        h_3 = self.block3(h_2)
        h_4 = self.block4(h_3)
        return h_1, h_2, h_3, h_4

class VGG19_Encoder(VGG19):
    def __init__(self):
        super(VGG19_Encoder, self).__init__()
    
    def forward(self, image):
        h_1 = self.block1(image)
        #print(h_1.shape)
        h_2 = self.block2(h_1)
        #print(h_2.shape)
        h_3 = self.block3(h_2)
        h_4 = self.block4(h_3)
        return h_4

class Reverse_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, pad_size=1):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size)
        
    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        return h

class VGG19_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Reverse_Layer(512, 256)
        self.layer2 = Reverse_Layer(256, 256)
        self.layer3 = Reverse_Layer(256, 256)
        self.layer4 = Reverse_Layer(256, 256)
        self.layer5 = Reverse_Layer(256, 128)
        self.layer6 = Reverse_Layer(128, 128)
        self.layer7 = Reverse_Layer(128, 64)
        self.layer8 = Reverse_Layer(64, 64)
        self.layer9 = Reverse_Layer(64, 3)
    def forward(self, features):
        h = features
        for i in range(1, 10):
            h = getattr(self, f'layer{i}')(h)
            if(i != 9):
                h = F.relu(h)
            if(i in [1, 5, 7]):
                h = F.interpolate(h, scale_factor=2)
        return h
        
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGG19_Encoder()
        self.decoder = VGG19_Decoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.layer = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x
        
class Generator_P(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super().__init__()

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 32 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 32 * 4 * 4),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 32, feature_dim * 16),
            self.dconv_bn_relu(feature_dim * 16, feature_dim * 8),
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),
            self.dconv_bn_relu(feature_dim * 2, feature_dim),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y

# Discriminator
class Discriminator_P(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator_P, self).__init__()

        self.l1 = nn.Sequential(
            self.conv_bn_lrelu(in_dim, feature_dim),
            self.conv_bn_lrelu(feature_dim, feature_dim),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),
            self.conv_bn_lrelu(feature_dim * 8, feature_dim * 16),
            nn.Conv2d(feature_dim * 16, 1, kernel_size=4, stride=1, padding=0),
        )
        self.apply(weights_init)

    def conv_bn_lrelu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.InstanceNorm2d(out_dim, affine = True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1, 1)
        return y

# setting for weight init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
