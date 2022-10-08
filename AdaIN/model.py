import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        features = models.vgg19(pretrained=True).features
        self.block1 = features[:2]    #VGG19 relu1_1
        self.block2 = features[2:7]   #VGG19 relu2_1
        self.block3 = features[7:12]  #VGG19 relu3_1
        self.block4 = features[12:21] #VGG19 relu4_1
#        for p in self.parameters():
#            p.requires_grad = False
        for i in range(1, 5):
            for param in getattr(self, f'block{i}').parameters():
                param.requires_grad = False
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

class DenseLayer(nn.Module):
    def __init__(self, input_features, growth_rate, bn_size=4, drop_rate=0):
        super(DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            #nn.InstanceNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_features, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.InstanceNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate != 0:
            y = self.dropout(y)
        return torch.cat([x, y], dim=1)

class TransitionLayer(nn.Module):
    def __init__(self, input, output):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            #nn.InstanceNorm2d(output),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.layer(x)

class ReverseDenseLayer(nn.Module):
    def __init__(self, input_features, output_features, growth_rate, bn_size=4):
        super(ReverseDenseLayer, self).__init__()
        self.dense_layer = nn.Sequential(
#            nn.Conv2d(in_channels=input_features, out_channels=bn_size * growth_rate, kernel_size=3, stride=1, padding=0, bias=False),
#            #nn.InstanceNorm2d(bn_size * growth_rate),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=output_features, kernel_size=1, stride=1, padding=1, bias=False),
#            #nn.InstanceNorm2d(input_features),
#            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_features, out_channels=output_features, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.InstanceNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dense_layer(x)

class ReverseTransitionLayer(nn.Module):
    def __init__(self, input, output):
        super(ReverseTransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(input, output, 4, 2, 1),
            #nn.InstanceNorm2d(output),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.layer(x)

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

class DenseNet_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        features = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True).features[:10]
        #features.apply(deactivate_batchnorm)
        self.layer1 = features[:4]
        self.layer2 = features[4:6]
        self.layer3 = features[6:8]
        self.layer4 = features[8:10]
        for i in range(1, 5):
            for param in getattr(self, f"layer{i}").parameters():
                param.requires_grad = False
    def forward(self, x):
        h_1 = self.layer1(x)
        h_2 = self.layer2(h_1)
        h_3 = self.layer3(h_2)
        h_4 = self.layer4(h_3)
        return h_4

class DenseNet_pretrained_Loss(DenseNet_pretrained):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        h_1 = self.layer1(x)
        h_2 = self.layer2(h_1)
        h_3 = self.layer3(h_2)
        h_4 = self.layer4(h_3)
        return h_1, h_2, h_3, h_4

class DenseNet_pretrained_Decoder(nn.Module):
    def __init__(self, start_features=64, num_layers=[6, 12, 24], growth_rate=32):
        super().__init__()
        features = (((start_features + num_layers[0] * growth_rate)//2 + num_layers[1] * growth_rate)//2 + num_layers[2] * growth_rate)//2
        self.trans1 = ReverseTransitionLayer(features, features * 2)
        features *= 2
        layer1 = [ReverseDenseLayer(features - i * growth_rate, features - (i + 1) * growth_rate, growth_rate) for i in range(num_layers[2])]
        self.block1 = nn.Sequential(*layer1)
        features -= num_layers[2] * growth_rate
        self.trans2 = ReverseTransitionLayer(features, features * 2)
        features *= 2
        layer2 = [ReverseDenseLayer(features - i * growth_rate, features - (i + 1) * growth_rate, growth_rate) for i in range(num_layers[1])]
        self.block2 = nn.Sequential(*layer2)
        features -= num_layers[1] * growth_rate
        self.trans3 = ReverseTransitionLayer(features, features * 2)
        features *= 2
        layer3 = [ReverseDenseLayer(features - i * growth_rate, features - (i + 1) * growth_rate, growth_rate) for i in range(num_layers[0])]
        self.block3 = nn.Sequential(*layer3)
        features -= num_layers[0] * growth_rate
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(start_features, start_features, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(start_features, 3, 7, 2, 3, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        #print("decoder:", x.shape)
        h_1 = self.trans1(x)
        #print("decoder:", h_1.shape)
        h_2 = self.block1(h_1)
        #print("decoder:", h_2.shape)
        h_2 = self.trans2(h_2)
        #print("decoder:", h_2.shape)
        h_3 = self.block2(h_2)
        #print("decoder:", h_3.shape)
        h_3 = self.trans3(h_3)
        #print("decoder:", h_3.shape)
        h_4 = self.block3(h_3)
        #print("decoder:", h_4.shape)
        h_4 = self.conv(h_4)
        #print("decoder:", h_4.shape)
        return h_4
    

class DenseNet_encoder(nn.Module):
    def __init__(self, start_features, growth_rate=32, num_layers=[6, 6, 24]):
        super(DenseNet_encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, start_features, 7, 2, 3, bias=False),
            #nn.InstanceNorm2d(start_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        features = start_features
        layer1 = [DenseLayer(features + i * growth_rate, growth_rate) for i in range(num_layers[0])]
        self.block1 = nn.Sequential(*layer1)
        features += num_layers[0] * growth_rate
        self.trans1 = TransitionLayer(features, features//2)
        features //= 2
        
        layer2 = [DenseLayer(features + i * growth_rate, growth_rate) for i in range(num_layers[1])]
        self.block2 = nn.Sequential(*layer2)
        features += num_layers[1] * growth_rate
        self.trans2 = TransitionLayer(features, features//2)
        features //= 2
        
        layer3 = [DenseLayer(features + i * growth_rate, growth_rate) for i in range(num_layers[2])]
        self.block3 = nn.Sequential(*layer3)
        features += num_layers[2] * growth_rate
        self.trans3 = TransitionLayer(features, features//2)
        features //= 2
    
    def forward(self, x):
        #print("encoder:", x.shape)
        h_1 = self.conv(x)
        #print("encoder:", h_1.shape)
        h_2 = self.block1(h_1)
        #print("encoder:", h_2.shape)
        h_2 = self.trans1(h_2)
        #print("encoder:", h_2.shape)
        h_3 = self.block2(h_2)
        #print("encoder:", h_3.shape)
        h_3 = self.trans2(h_3)
        #print("encoder:", h_3.shape)
#        h_4 = self.block3(h_3)
#        #print("encoder:", h_4.shape)
#        h_4 = self.trans3(h_4)
#        #print("encoder:", h_4.shape)
        return h_3

class DenseNet_decoder(nn.Module):
    def __init__(self, start_features, growth_rate=32, num_layers=6):
        super(DenseNet_decoder, self).__init__()
        self.trans1 = ReverseTransitionLayer(((start_features + num_layers * growth_rate)//2 + num_layers * growth_rate)//2, (start_features + num_layers * growth_rate)//2 + num_layers * growth_rate)
        layer1 = [ReverseDenseLayer((start_features + num_layers * growth_rate)//2 + num_layers * growth_rate - i * growth_rate, (start_features + num_layers * growth_rate)//2 + num_layers * growth_rate - (i + 1) * growth_rate, growth_rate) for i in range(num_layers)]
        self.block1 = nn.Sequential(*layer1)
        self.trans2 = ReverseTransitionLayer((start_features + num_layers * growth_rate)//2, start_features + num_layers * growth_rate)
        layer2 = [ReverseDenseLayer(start_features + num_layers * growth_rate - i * growth_rate, start_features + num_layers * growth_rate - (i + 1) * growth_rate, growth_rate) for i in range(num_layers)]
        self.block2 = nn.Sequential(*layer2)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(start_features, start_features, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(start_features, 3, 7, 2, 3, 1, bias=False),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        #print("decoder:", x.shape)
        h_1 = self.trans1(x)
        #print("decoder:", h_1.shape)
        h_2 = self.block1(h_1)
        #print("decoder:", h_2.shape)
        h_2 = self.trans2(h_2)
        #print("decoder:", h_2.shape)
        h_3 = self.block2(h_2)
        #print("decoder:", h_3.shape)
        h_3 = self.conv(h_3)
        #print("decoder:", h_3.shape)
        return h_3
        
        
def cal_mu_std(features):
    batch_size, c = features.size()[:2]
    features_mu = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mu, features_std
    
def cal_content_loss(f_t, t):
    return F.mse_loss(f_t, t)
    
def cal_style_loss(g_t, t):
    loss = 0
    for i, j in zip(g_t, t):
        g_mu, g_std = cal_mu_std(i)
        t_mu, t_std = cal_mu_std(j)
        loss += F.mse_loss(g_mu, t_mu) + F.mse_loss(g_std, t_std)
    return loss

def adain(content_encoded, style_encoded):
    content_mu, content_std = cal_mu_std(content_encoded)
    style_mu, style_std = cal_mu_std(style_encoded)
    #print(content_mu, content_std)
    return style_std*(content_encoded - content_mu)/content_std + style_mu

class AdaIN_Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if model == "vgg":
            self.encoder = VGG19()
            self.decoder = VGG19_Decoder()
            #self.Loss = self.encoder
        if model == "dense":
            self.encoder = DenseNet_encoder(64)
            self.decoder = DenseNet_decoder(64)
            #self.encoder = DenseNet_pretrained()
            #self.decoder = DenseNet_pretrained_Decoder()
            self.Loss = VGG19()
            #self.Loss = DenseNet_pretrained_Loss()
        
    def forward(self, content_images, style_images):
        content_encoded = self.encoder(content_images)
        style_encoded = self.encoder(style_images)
        
        #計算Loss
        if self.model == "vgg":
            t = adain(content_encoded[-1], style_encoded[-1])
        
        if self.model == "dense":
            t = adain(content_encoded, style_encoded)
        
        output = self.decoder(t)
        
        if self.model == "dense":
            #out_content = self.decoder(content_encoded)
            content_images = normalize(content_images)
            style_images = normalize(style_images)
            output = normalize(output)
            content_features = self.Loss(content_images)
            style_features = self.Loss(style_images)
            output_features = self.Loss(output)
            t = adain(content_features[-1], style_features[-1])
            #content_loss = F.mse_loss(out_content, content_images)
            
        if self.model == "vgg":
            style_features = style_encoded
            output_features = self.encoder(output)
        #print(output_features[-1].shape, t.shape)
        content_loss = cal_content_loss(output_features[-1], t)
        style_loss = cal_style_loss(output_features, style_features)
        #print(content_loss, style_loss)
        return content_loss, style_loss
    
    def generate(self, content_image, style_image):
        if self.model == "vgg":
            content_encoded = self.encoder(content_image)[-1]
            style_encoded = self.encoder(style_image)[-1]
        if self.model == "dense":
            content_encoded = self.encoder(content_image)
            style_encoded = self.encoder(style_image)
            
        t = adain(content_encoded, style_encoded)
        output = self.decoder(t)
        return output
    
    def origin(self, content_image):
        if self.model == "vgg":
            content_encoded = self.encoder(content_image)[-1]
        if self.model == "dense":
            content_encoded = self.encoder(content_image)
        output = self.decoder(content_encoded)
        return output
