import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout,BatchNorm2d
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder
import math

ic=200
oc=256
ws=13
fs=(ws+1)//2
png_in=fs*fs

tc=oc
L1O=128

num_class=16

d_model=ws*ws

def position_embeddings(n_pos_vec, dim):
    position_embedding = torch.nn.Embedding(n_pos_vec.numel(), dim)
    torch.nn.init.constant_(position_embedding.weight, 0.)
    return position_embedding

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_convs=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_convs(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up,self).__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.up_match_channels=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x1,x2):
        x1=self.up(x1)
        x1=self.up_match_channels(x1)

        diffY=x2.size()[2]-x1.size()[2]
        diffX= x2.size()[3]-x1.size()[3]

        x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])

        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x):
        return self.conv(x)

class MambaLayers(nn.Module):
    def __init__(self, d_model, n_layers):
        super(MambaLayers, self).__init__()
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x

class MambaLayersRev(nn.Module):
    def __init__(self, d_model, n_layers):
        super(MambaLayersRev, self).__init__()
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)

        x=torch.flip(x, dims=[1])
        x = self.mamba(x)
        x=torch.flip(x, dims=[1])

        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x

def generate_spiral_indices(H, device="cpu"):
    """
    Generates spiral-unrolled coordinates (i, j) and returns indices on the same device as the input tensor.
    """
    indices = []
    visited = [[False] * H for _ in range(H)]
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    x, y = H // 2, H // 2  # Start from the center
    step = 1
    d = 0  # Initial direction: right

    while len(indices) < H * H:
        for _ in range(2):  # Each step goes twice (two edges per loop)
            for _ in range(step):
                if 0 <= x < H and 0 <= y < H and not visited[x][y]:
                    visited[x][y] = True
                    indices.append((x, y))
                x += dirs[d][0]
                y += dirs[d][1]
            d = (d + 1) % 4  # Change direction
        step += 1
    
    # Return indices on the same device as the input tensor
    i_indices = torch.tensor([i for i, j in indices], device=device)
    j_indices = torch.tensor([j for i, j in indices], device=device)
    return i_indices, j_indices


def spiral_flatten(x):
    """
    x: Tensor of shape (B, C, H, W), H == W
    returns: Tensor of shape (B, C, H*W), flattened in spiral order
    """
    B, C, H, W = x.shape
    assert H == W, "Only square inputs (H == W) are supported"
    
    # Generate spiral-ordered indices (automatically matches the device of x)
    i_indices, j_indices = generate_spiral_indices(H, device=x.device)
    
    # Use torch.gather to collect data (avoid explicit indexing)
    x_flat = x.reshape(B, C, H * W)  # (B, C, H*W)
    idx = i_indices * H + j_indices  # Calculate linear index (H*W,)
    x_spiral = torch.gather(x_flat, 2, idx.expand(B, C, -1))  # (B, C, H*W)
    return x_spiral


def spiral_unflatten(x_spiral, H):
    """
    x_spiral: Tensor of shape (B, C, H*W), flattened in spiral order
    H: Height and width after restoration
    returns: Tensor of shape (B, C, H, W), restored to original shape
    """
    B, C, L = x_spiral.shape
    assert H * H == L, "The length of the input tensor does not meet the requirement of H×H"
    
    # Generate spiral-ordered indices (automatically matches the device of x_spiral)
    i_indices, j_indices = generate_spiral_indices(H, device=x_spiral.device)
    
    # Use scatter_ to restore data
    x_restored = torch.zeros(B, C, H, H, device=x_spiral.device)
    idx = i_indices * H + j_indices  # Linear index (H*W,)
    x_restored = x_restored.view(B, C, H * H)
    x_restored.scatter_(2, idx.expand(B, C, -1), x_spiral)  # Fill data
    return x_restored.view(B, C, H, H)

class MambaLayersCenter(nn.Module):
    def __init__(self, d_model, n_layers):
        super(MambaLayersCenter, self).__init__()
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))

    def forward(self, x):
        bs, c, h, w = x.shape

        x=spiral_flatten(x)
        x=x.permute(0,2,1)
        x = self.mamba(x)
        x=x.permute(0,2,1)
        spiral_unflatten(x,h) 

        x = x.reshape(bs, c, h, w)
        return x

class EncoderLayers(nn.Module):
    def __init__(self,encoder_in=ic,num_encoder_layers=3,dim_feedforward=384,nhead=8,dropout=0.1):
        super(EncoderLayers, self).__init__()
        encoder_layer = TransformerEncoderLayer(encoder_in, nhead, dim_feedforward, dropout,norm_first=False)
        encoder_norm =LayerNorm(encoder_in)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x    

class Pooling(nn.Module):
    """
    @article{ref-vit,
    title={An image is worth 16x16 words: Transformers for image recognition at scale},
    author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, 
            Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
    journal={arXiv preprint arXiv:2010.11929},
    year={2020}
    }
    """
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  

class ChannelAttn(nn.Module):
    def __init__(self,encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8,dropout=0.1):
        super(ChannelAttn, self).__init__()
        encoder_layer = TransformerEncoderLayer(encoder_in,nhead,dim_feedforward,dropout=0.1,norm_first=False)
        self.encoder0 = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=LayerNorm(ic))

    def forward(self, x):
        batch_size, c, h, w = x.shape
        center_h = (h - 1) // 2
        center_w = (w - 1) // 2

        values = x[:, :, center_h, center_w].unsqueeze(1)
        values=self.encoder0(values).flatten() # shape: (batch_size * c, )

        h_idx = torch.full((batch_size * c,), center_h, device=x.device, dtype=torch.long)
        w_idx = torch.full((batch_size * c,), center_w, device=x.device, dtype=torch.long)

        batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(c)
        channel_idx = torch.arange(c, device=x.device).repeat(batch_size)

        x = torch.index_put(x, (batch_idx, channel_idx, h_idx, w_idx), values, accumulate=False)

        return x  
    
class MyModel(torch.nn.Module):
    def __init__(self,batch_size=32,bilinear=True):
        super(MyModel, self).__init__()
        
        self.inc= DoubleConv(ic,oc)
        self.down1= Down(oc,512)
        self.down2=Down(512,1024)

        self.up2= Up(512,oc,bilinear)
        self.up1= Up(1024,512,bilinear)

        self.dropout=Dropout(0.5)

        self.mamba = MambaLayers(d_model=ic,n_layers=1)
        self.mamba2 = MambaLayersCenter(d_model=ic,n_layers=1)

        self.encoder0 = ChannelAttn(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8,dropout=0.1)
        self.encoder1 = EncoderLayers(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8)
        
        self.position_embedding = position_embeddings(torch.arange(batch_size*ic*ws*ws), 1)  
        self.BN=nn.BatchNorm2d(ic)
        self.lmd = nn.Parameter(torch.tensor(0.5))

        self.pool = Pooling(pool="mean")
        self.classifier = Classifier(dim=oc, num_classes=num_class)

    def forward(self, x):
        batch_size, c, h, w= x.shape
        x1=self.mamba2(x)
        x=self.mamba(x)
        x-(1-self.lmd)*x+self.lmd*x1
        x=self.BN(x)
        
        x1=self.encoder0(x)
        
        position_ids = torch.arange(batch_size*ic*ws*ws, dtype=torch.long, device=x.device)  
        position_embeds = self.position_embedding(position_ids).reshape(batch_size, ic,ws,ws) 
        x=x+position_embeds

        x=self.encoder1(x)
        
        x=x+x1

        x1= self.inc(x)
        
        x2= self.down1(x1)

        x3= self.down2(x2)
      
        x= self.up1(x3,x2)

        x=self.up2(x,x1)
        x=self.dropout(x)

        x=x.reshape(batch_size,-1,h*w)

        x=x.permute(0,2,1)

        x=self.pool(x)

        x = self.classifier(x)

        return x

if __name__=='__main__':
    print('start')
    batch_size=32
    model = MyModel(batch_size=batch_size, bilinear=True)

    # (batch_size, ic, height, width)
    dummy_input = torch.randn(batch_size, ic, 13, 13) 

    output = model(dummy_input)

    print("Output shape:", output.shape)