"""
UNet 3+ implementation in PyTorch,
based on https://doi.org/10.48550/arXiv.2004.08790

Version 0.1.0
"""

import torch
from torch import nn
from typing import List, Tuple
from collections import OrderedDict
from torchvision.models import resnet101

class UNetConvBlock(nn.Module):
    """A convolutional block in UNet"""
    def __init__(self, input_channels: int, 
               output_channels: int,
               num_convs: int = 2,
               kernel_size:int = 3,
               padding:int = 1,
               stride:int = 1,
               is_batchnorm: bool = True):
        super(UNetConvBlock, self).__init__()
    
        layers = []
        
        if is_batchnorm:
            for _ in range(num_convs):
                layers.extend([
                    nn.Conv2d(in_channels = input_channels,
                        out_channels = output_channels,
                        stride = stride,
                        padding = padding,
                        kernel_size = kernel_size),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace= True)
                ])

                input_channels = output_channels
        else:
            for _ in range(num_convs):
                layers.extend([
                    nn.Conv2d(in_channels = input_channels,
                        out_channels = output_channels,
                        stride = stride,
                        padding = padding,
                        kernel_size = kernel_size),
                    nn.ReLU(inplace= True)
                ])

                input_channels = output_channels
                
        self.layers = nn.Sequential(*layers)
        
        # Weights are initialized using kaming uniform distribution according to https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L182
    
    def forward(self, x):
        return self.layers(x)

class UNetUpBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 bilinear: bool = False):
        super(UNetUpBlock, self).__init__()
        
        self.convblock = UNetConvBlock(in_channels, out_channels)
        
        if bilinear:
            self.upsample = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0, stride = 1) # Halves the number of channels
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size = 4, stride = 2,
                                               padding = 1)
            
    def forward(self, x, *maps):
        x = self.upsample(x)
        
        for i in range(len(maps)):
            x = torch.cat([x, maps[i]], 1)
            
        return self.convblock(x)
    

class UpConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, is_batchnorm = True):
        super(UpConvBnReLU, self).__init__()
        
        if is_batchnorm:
            self.layers = nn.Sequential(
                nn.Upsample(scale_factor = scale_factor, mode = "bilinear"),
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = True)
            )
        else:
            self.layers = nn.Sequential(
                nn.Upsample(scale_factor = scale_factor, mode = "bilinear"),
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True)
            )
        
        
    def forward(self, x):
        return self.layers(x)
    
    
class MaxConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool_ks = None, is_batchnorm = True):
        super(MaxConvBnReLU, self).__init__()
        
        layers = []
        if max_pool_ks:
            layers = [nn.MaxPool2d(kernel_size = max_pool_ks, stride = max_pool_ks, ceil_mode = True)]
        
        if is_batchnorm:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = True)
            ])
        else:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True)
            ])
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    


class UNetEncoder(nn.Module):
    def __init__(self, channels: List[int] = [3, 64, 128, 256, 512, 1024], 
                 block_padding: int = 1,
                 is_batchnorm: bool = True):
        super(UNetEncoder, self).__init__()

        self.enc_blocks = nn.ModuleList([
            UNetConvBlock(channels[i], channels[i+1], padding = block_padding, is_batchnorm = is_batchnorm) 
            for i in range(len(channels) - 1)  
        ])
        self.max_pool = nn.MaxPool2d(kernel_size = 2,
                                     stride = 2,
                                     padding = 0)
        
    def forward(self, x):
        enc_outputs = []

        for block in self.enc_blocks:
            x = block(x)
            enc_outputs.append(x)
            x = self.max_pool(x)

        return enc_outputs #, x # x = The max pooled output of the last block
    

class ResNet101Encoder(nn.Module):
    def __init__(self, weights = None):
        super(ResNet101Encoder, self).__init__()
        self.resnet = resnet101(weights = weights, progress = False)
        
    def forward(self, x):
        enc_outputs = []
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        
        enc_outputs.append(self.resnet.relu(x))
        enc_outputs.append(self.resnet.layer1(self.resnet.maxpool(enc_outputs[-1])))
        enc_outputs.append(self.resnet.layer2(enc_outputs[-1]))
        enc_outputs.append(self.resnet.layer3(enc_outputs[-1]))
        enc_outputs.append(self.resnet.layer4(enc_outputs[-1]))
        
        return enc_outputs


class UNet3PlusDecoder(nn.Module):
    def __init__(self, n_classes: int,
                 class_guided = True,
                 channels: List[int] = [64, 128, 256, 512, 1024], 
                 block_padding: int = 1,
                 is_batchnorm: bool = True):
        super(UNet3PlusDecoder, self).__init__()
        
        self.class_guided = class_guided
        
        catChannels = channels[0] # Channels number of each map in the full-scale aggregated feature map
        upChannels = catChannels * len(channels) # Total number of channels in the full-scale aggregated feature map
        
        dec_blocks = OrderedDict() # Decoder blocks
        num_decoders = len(channels) - 1
        
        for i in range(num_decoders): # Iterate over decoder blocks starting from deepest one ..., D4, D3, D2, D1
           
            # Each decoder gets 5 inputs in the case of default channels
            
            intra_connections = OrderedDict() # Intra skip connections between current decoder block and it's preceding decoder blocks
            inter_connections = OrderedDict() # Inter skip connections between current decoder block and it's corresponding encoder blocks
            
            num_prev_decoders = i + 1
            num_prev_encoders = num_decoders - i
            
            for j in range(num_prev_encoders): # Iterate over previous encoder blocks starting from the first to the parallel one 
                max_pool_kernel_size = None if j == (num_prev_encoders - 1) else 2**(num_prev_encoders - 1 - j) # ..., 8, 4, 2
                inter_connections[f"con_enc_{j}_dec_{num_decoders - 1 - i}"] = MaxConvBnReLU(in_channels = channels[j], 
                                                                                         out_channels = catChannels, 
                                                                                         max_pool_ks = max_pool_kernel_size,
                                                                                         is_batchnorm = is_batchnorm)
                
            
            for j in range(num_prev_encoders, len(channels)): # Iterate over previous decoder blocks starting from the previous to deepest one
                in_channels = channels[-1] if j == num_decoders else upChannels
                scale_factor = 2**(j - num_prev_encoders + 1) # 2, 4, 8, 16, ...
                intra_connections[f"con_dec_{j}_dec_{num_decoders - 1 - i}"] = UpConvBnReLU(in_channels = in_channels, 
                                                                                        out_channels = catChannels, 
                                                                                        scale_factor = scale_factor,
                                                                                        is_batchnorm = is_batchnorm)
                
                  
            # The conv layer for applying on fused feature maps
            aggregator = MaxConvBnReLU(in_channels = upChannels, out_channels = upChannels, is_batchnorm = is_batchnorm) # Max pool is disabled by default
            
            dec_blocks[f"dec_block_{num_decoders - i - 1}"] = nn.ModuleDict(OrderedDict(**inter_connections, **intra_connections, **{f"dec_{i}_aggregator": aggregator}))
            
        self.dec_blocks = nn.ModuleDict(dec_blocks)
        

        # Implement up sampling and convolution layers for deep supervision
        deepsup_blocks = []
        for i in range(len(self.dec_blocks), -1, -1): # Put -1 as lower bound to include the last encoder block (deepest one)
            if i == 0:
                deepsup_blocks.append(nn.Conv2d(upChannels, n_classes, kernel_size = 3, padding = 1))
            else:
                deepsup_blocks.append(nn.Sequential(
                        nn.Conv2d(channels[-1] if i == len(self.dec_blocks) else upChannels, n_classes, kernel_size = 3, padding = 1),
                        nn.Upsample(scale_factor = 2**i, mode = "bilinear") # scale_factor = ..., 16, 8, 4, 2
                    ))
        self.deepsup_blocks = nn.ModuleList(deepsup_blocks)
        
        if self.class_guided:
            # Classification layer for applying on final encoder block features
            self.deep_classifier = nn.Sequential(
                nn.Dropout(p = 0.5),
                nn.Conv2d(channels[-1], 2, kernel_size = 1),
                nn.AdaptiveMaxPool2d(1),
                nn.Sigmoid()
            )
        
        
    def _dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final
    
            
    def forward(self, prev_outputs):
        # Here prev_outputs contain encoder outputs from first to deepest
        
        outputs_dict = {}
        
        if self.class_guided:
            # Classification of final output of the encoder
            # deep_classifier predicts whether a sample has object or not no matter what classes of objects it might have (ToDo: make this layer
            # predicting for each specific object class)
            outputs_dict["cls"] = self.deep_classifier(prev_outputs[-1]).squeeze([-1, -2]) # (B,N,1,1)->(B,N)
            class_preds = outputs_dict["cls"].argmax(dim = 1)
            class_preds = class_preds.unsqueeze(-1).float()
        
        # Decoder blocks
        for i, block in enumerate(self.dec_blocks.values()):
            block_input_maps = []
            for j, enc_dec_con in enumerate(block.values()): # Iter and Intra connections followed by aggregator
                if j == (len(block)-1): # Concat the feature maps and pass them through aggregator
                    prev_outputs[len(block)-3-i] = enc_dec_con(torch.cat(block_input_maps, dim=1)) # Substitute the output of encoder block parallel with the current decoder block with cuuernt output
                else:
                    block_input_maps.append(enc_dec_con(prev_outputs[j]))
                    
        # Here prev_outputs contain decoder outputs from first to deepest
        # Pass them through conv and upsampling to produce intermidiate and final feature maps
        prev_outputs = prev_outputs[::-1] # deepest to first
        for i in range(len(prev_outputs)):
            if i == len(prev_outputs) - 1:
                outputs_dict["final_pred"] = self.deepsup_blocks[i](prev_outputs[i])
            else:
                outputs_dict[f"aux_head{i}"] = self.deepsup_blocks[i](prev_outputs[i])
                
        if self.class_guided:
            # Multiply predicted masks with predicted class label
            for key in outputs_dict:
                if key != "cls":
                    outputs_dict[key] = self._dotProduct(outputs_dict[key], class_preds) # ToDo: change to *
            
        return outputs_dict # Predicted masks (auxiliaries, head), Predicted class
        

class UNet3Plus(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 enc_channels: List[int] = [3, 64, 128, 256, 512, 1024],
                 dec_channels: List[int] = [64, 128, 256, 512, 1024], 
                 class_guided: bool = True,
                 is_batchnorm: bool = True):
        super(UNet3Plus, self).__init__()
        self.encoder = UNetEncoder(channels = enc_channels, is_batchnorm = is_batchnorm)
        self.decoder = UNet3PlusDecoder(channels = dec_channels, n_classes = num_classes, class_guided = class_guided, is_batchnorm = is_batchnorm)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    

class ResNet101UNet3Plus(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 resnet_weights: None,
                 class_guided: bool = True,
                 is_batchnorm: bool = True,
                 output_size: Tuple[int, ...] = None):
        super(ResNet101UNet3Plus, self).__init__()
        self.encoder = ResNet101Encoder(weights = resnet_weights)
        self.decoder = UNet3PlusDecoder(channels = [64, 256, 512, 1024, 2048], n_classes = num_classes, class_guided = class_guided, is_batchnorm = is_batchnorm)
        self.output_size = output_size
        
    def forward(self, x):
        outputs_dict = self.decoder(self.encoder(x))
        
        if self.output_size:
            upsample = nn.Upsample(size = self.output_size, mode = "bilinear")
            for k in outputs_dict:
                outputs_dict[k] = upsample(outputs_dict[k])
                
        return outputs_dict

class ResNet101UNet3Plus_SingleOut(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 resnet_weights: None,
                 class_guided: bool = True,
                 is_batchnorm: bool = True,
                 output_size: Tuple[int, ...] = None):
        super(ResNet101UNet3Plus_SingleOut, self).__init__()
        self.encoder = ResNet101Encoder(weights = resnet_weights)
        self.decoder = UNet3PlusDecoder(channels = [64, 256, 512, 1024, 2048], n_classes = num_classes, class_guided = class_guided, is_batchnorm = is_batchnorm)
        self.output_size = output_size
        
    def forward(self, x):
        outputs_dict = self.decoder(self.encoder(x))
        
        if self.output_size:
            upsample = nn.Upsample(size = self.output_size, mode = "bilinear")
            for k in outputs_dict:
                outputs_dict[k] = upsample(outputs_dict[k])
                
        return outputs_dict["final_pred"][:, 1]