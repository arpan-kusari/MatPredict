import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        in_channels: RGB image, so it's 3 channels 
        out_channels: 3 since the basecolor is also RGB image 
        """
        super(UNet, self).__init__()

        #incoder, extract the feature vector
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B,64,H,W]
        p1 = self.pool(e1) # [B,64,H/2,W/2]

        e2 = self.enc2(p1) # [B,128,H/2,W/2]
        p2 = self.pool(e2) # [B,128,H/4,W/4]

        e3 = self.enc3(p2) # [B,256,H/4,W/4]
        p3 = self.pool(e3) # [B,256,H/8,W/8]

        e4 = self.enc4(p3) # [B,512,H/8,W/8]
        p4 = self.pool(e4) # [B,512,H/16,W/16]

        # Bottleneck
        b = self.bottleneck(p4) # [B,1024,H/16,W/16]

        # Decoder
        up4 = self.up4(b)               # [B,512,H/8,W/8]
        merge4 = torch.cat([up4, e4], dim=1) # [B,1024,H/8,W/8]
        d4 = self.dec4(merge4) # [B,512,H/8,W/8]

        up3 = self.up3(d4)             # [B,256,H/4,W/4]
        merge3 = torch.cat([up3, e3], dim=1) # [B,512,H/4,W/4]
        d3 = self.dec3(merge3) # [B,256,H/4,W/4]

        up2 = self.up2(d3) # [B,128,H/2,W/2]
        merge2 = torch.cat([up2, e2], dim=1) # [B,256,H/2,W/2]
        d2 = self.dec2(merge2) # [B,128,H/2,W/2]

        up1 = self.up1(d2) # [B,64,H,W]
        merge1 = torch.cat([up1, e1], dim=1)# [B,128,H,W]
        d1 = self.dec1(merge1) # [B,64,H,W]

        out = self.out_conv(d1)  # [B, out_channels, H, W]

        return out



class UNet_without_overlap(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        in_channels: RGB image, so it's 3 channels 
        out_channels: 3 since the basecolor is also RGB image 
        """
        super(UNet_without_overlap, self).__init__()

        #incoder, extract the feature vector
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B,64,H,W]
        p1 = self.pool(e1) # [B,64,H/2,W/2]

        e2 = self.enc2(p1) # [B,128,H/2,W/2]
        p2 = self.pool(e2) # [B,128,H/4,W/4]

        e3 = self.enc3(p2) # [B,256,H/4,W/4]
        p3 = self.pool(e3) # [B,256,H/8,W/8]

        e4 = self.enc4(p3) # [B,512,H/8,W/8]
        p4 = self.pool(e4) # [B,512,H/16,W/16]

        # Bottleneck
        b = self.bottleneck(p4) # [B,1024,H/16,W/16]

        # Decoder
        up4 = self.up4(b)               # [B,512,H/8,W/8]
        # merge4 = torch.cat([up4, e4], dim=1) # [B,1024,H/8,W/8]
        d4 = self.dec4(up4) # [B,512,H/8,W/8]

        up3 = self.up3(d4)             # [B,256,H/4,W/4]
        # merge3 = torch.cat([up3, e3], dim=1) # [B,512,H/4,W/4]
        d3 = self.dec3(up3) # [B,256,H/4,W/4]

        up2 = self.up2(d3) # [B,128,H/2,W/2]
        # merge2 = torch.cat([up2, e2], dim=1) # [B,256,H/2,W/2]
        d2 = self.dec2(up2) # [B,128,H/2,W/2]

        up1 = self.up1(d2) # [B,64,H,W]
        # merge1 = torch.cat([up1, e1], dim=1)# [B,128,H,W]
        d1 = self.dec1(up1) # [B,64,H,W]

        out = self.out_conv(d1)  # [B, out_channels, H, W]

        return out



#ResNet 50 
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
    
        if backbone == 'resnet50':
            net = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            net = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet18':
            net = models.resnet18(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 2. remove the fc layer and the avgpool layer 
        #    ResNet structure 
        #    [conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc]
    
        self.features = nn.Sequential(*list(net.children())[:-2])
        
    def forward(self, x):
        # 3. forward result is: flattened feature: 
        #    这里输出 shape: [batch_size, 2048, 1, 1] (以resnet50为例)
        #    如果你不想做全局平均池化，则可以把[:-1]改成[:-2]，这样就保留到layer4的输出(7x7特征图)。
        out = self.features(x)
        return out

 


#decoder 
class SimpleDecoder(nn.Module):
 
    def __init__(self, in_channels=2048, out_channels=3):
        super().__init__()

        # 1st up sample: 2048 -> 1024, space x2
        self.up4 = nn.ConvTranspose2d(in_channels, 1024, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2nd up sample: 512 -> 256, space x2
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 3rd up sample: 128 -> 64, space x2
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 4th up sample: 32 -> 16, space x2
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # the shape change:  H/32 -> H/16 -> H/8 -> H/4 -> H/2 -> H
        
        # output layer:
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1) # [batchsize, 3, 224, 224]

        self.up0 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)

    def forward(self, x4):
        # x4: [B, 2048, H/32, W/32]
        x = self.up4(x4)   # -> [B,1024, H/16, W/16]
        x = self.dec4(x)   # -> [B, 512, H/16, W/16]

        x = self.up3(x)    # -> [B, 256, H/8, W/8]
        x = self.dec3(x)   # -> [B, 128, H/8, W/8]

        x = self.up2(x)    # -> [B, 64, H/4, W/4]
        x = self.dec2(x)   # -> [B, 32, H/4, W/4]

        x = self.up1(x)    # -> [B, 16, H/2, W/2]
        x = self.dec1(x)   # -> [B, 16, H/2, W/2]

      
         
        x = self.up0(x)         # -> [B,16, H, W]

        out = self.out_conv(x)  # -> [B, out_channels, H, W]
        return out
    



class Resnet50_combined(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, out_channels=3):
        super().__init__()
        #get encoder 
        self.encoder = ResNetFeatureExtractor(backbone=backbone, pretrained=pretrained)
        #get decoder 
        if backbone == 'resnet50':
            in_channels = 2048
        elif backbone == 'resnet18':
            in_channels = 512
        else:
            raise ValueError("Unsupported backbone")

        self.decoder = SimpleDecoder(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        feats = self.encoder(x)  # [B, in_channels, H/32, W/32] for resnet50
        out = self.decoder(feats)  # [B, out_channels, H, W]
        return out
    

from torchvision.models import efficientnet_b0
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = efficientnet_b0(pretrained=pretrained)
    
        self.features = net.features

    def forward(self, x):
        return self.features(x)

class EfficientNetDecoder(nn.Module):
    def __init__(self, in_channels=1280, out_channels=3):
        super().__init__()
        # 5次上采样: 7->14->28->56->112->224
        self.up1 = nn.ConvTranspose2d(in_channels, 640, kernel_size=2, stride=2)  # 7 -> 14
        self.dec1 = nn.Sequential(
            nn.Conv2d(640, 640, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(640, 320, kernel_size=2, stride=2)  # 14 -> 28
        self.dec2 = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(320, 160, kernel_size=2, stride=2)  # 28 -> 56
        self.dec3 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2)   # 56 -> 112
        self.dec4 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2)    # 112 -> 224
        self.out_conv = nn.Conv2d(40, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up3(x)
        x = self.dec3(x)
        x = self.up4(x)
        x = self.dec4(x)
        x = self.up5(x)
        out = self.out_conv(x)
        return out

class EfficientNet_combined(nn.Module):
    def __init__(self, pretrained=True, out_channels=3):
        super().__init__()
        self.encoder = EfficientNetFeatureExtractor(pretrained=pretrained)
        self.decoder = EfficientNetDecoder(in_channels=1280, out_channels=out_channels)

    def forward(self, x):
        feats = self.encoder(x)  # [B, 1280, 7, 7]
        out = self.decoder(feats)  
        return out
    



 
from torchvision.models import densenet121

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = densenet121(pretrained=pretrained)
       
        self.features = net.features

    def forward(self, x):
        return self.features(x)

class DenseNetDecoder(nn.Module):
    def __init__(self, in_channels=1024, out_channels=3):
        super().__init__()
        # 5次上采样: 7->14->28->56->112->224
        self.up1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2)  # 7 -> 14
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 14 -> 28
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 28 -> 56
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 56 -> 112
        self.dec4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # 112 -> 224
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up3(x)
        x = self.dec3(x)
        x = self.up4(x)
        x = self.dec4(x)
        x = self.up5(x)
        out = self.out_conv(x)
        return out

class DenseNet_combined(nn.Module):
    def __init__(self, pretrained=True, out_channels=3):
        super().__init__()
        self.encoder = DenseNetFeatureExtractor(pretrained=pretrained)
        self.decoder = DenseNetDecoder(in_channels=1024, out_channels=out_channels)

    def forward(self, x):
        feats = self.encoder(x)  # [B, 1024, 7, 7]
        out = self.decoder(feats)   
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================
# 1. based Swin Transformer model
# =======================
import timm

class SwinTransformerFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
     
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True)
    
    def forward(self, x):
        features = self.model(x)
 
        feats =  features[-1]  
        return feats.permute(0, 3, 1, 2)

class SwinTransformerDecoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=3):
        super().__init__()
  
        self.up1 = nn.ConvTranspose2d(in_channels, 384, kernel_size=2, stride=2)  # 7 -> 14
        self.dec1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)  # 14 -> 28
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)   # 28 -> 56
        self.dec3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)    # 56 -> 112
        self.dec4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)    # 112 -> 224
        self.out_conv = nn.Conv2d(24, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.up1(x)   # [B,384,14,14]
        x = self.dec1(x)  # [B,384,14,14]
        x = self.up2(x)   # [B,192,28,28]
        x = self.dec2(x)  # [B,192,28,28]
        x = self.up3(x)   # [B,96,56,56]
        x = self.dec3(x)  # [B,96,56,56]
        x = self.up4(x)   # [B,48,112,112]
        x = self.dec4(x)  # [B,48,112,112]
        x = self.up5(x)   # [B,24,224,224]
        out = self.out_conv(x)  # [B,3,224,224]
        return out

class SwinTransformerCombined(nn.Module):
    def __init__(self, pretrained=True, out_channels=3):
        super().__init__()
        self.encoder = SwinTransformerFeatureExtractor(pretrained=pretrained)
        self.decoder = SwinTransformerDecoder(in_channels=768, out_channels=out_channels)
    
    def forward(self, x):
        feats = self.encoder(x)  
        out = self.decoder(feats)  
        return out

# =======================
# 2. based on ConvNeXt model
# =======================
from torchvision.models import convnext_tiny

class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = convnext_tiny(pretrained=pretrained)
 
        self.backbone = nn.Sequential(*list(model.children())[:-2])
    
    def forward(self, x):
        feats = self.backbone(x)
        return feats

class ConvNeXtDecoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=3):
        super().__init__()
 
        self.up1 = nn.ConvTranspose2d(in_channels, 384, kernel_size=2, stride=2)  # 7 -> 14
        self.dec1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)  # 14 -> 28
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)   # 28 -> 56
        self.dec3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)    # 56 -> 112
        self.dec4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)    # 112 -> 224
        self.out_conv = nn.Conv2d(24, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.up1(x)   # [B,384,14,14]
        x = self.dec1(x)  # [B,384,14,14]
        x = self.up2(x)   # [B,192,28,28]
        x = self.dec2(x)  # [B,192,28,28]
        x = self.up3(x)   # [B,96,56,56]
        x = self.dec3(x)  # [B,96,56,56]
        x = self.up4(x)   # [B,48,112,112]
        x = self.dec4(x)  # [B,48,112,112]
        x = self.up5(x)   # [B,24,224,224]
        out = self.out_conv(x)  # [B,3,224,224]
        return out

class ConvNeXtCombined(nn.Module):
    def __init__(self, pretrained=True, out_channels=3):
        super().__init__()
        self.encoder = ConvNeXtFeatureExtractor(pretrained=pretrained)
        self.decoder = ConvNeXtDecoder(in_channels=768, out_channels=out_channels)
    
    def forward(self, x):
        feats = self.encoder(x)   
        out = self.decoder(feats)  
        return out

 
