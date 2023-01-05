# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import List, Tuple

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerLayer(nn.Module):
    def __init__(self, attn, ff) -> None:
        super().__init__()
        self.attn = attn
        self.ff = ff
    
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerLayer(
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncodingLayer(nn.Module):
    def __init__(self, to_patch_embedding, pos_embedding, cls_token, dropout) -> None:
        super().__init__()
        self.to_patch_embedding = to_patch_embedding
        self.pos_embedding = pos_embedding
        self.cls_token = cls_token
        self.dropout = dropout
    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        return x

class Classifier(nn.Module):
    def __init__(self, pool, to_latent, mlp_head) -> None:
        super().__init__()
        self.pool = pool
        self.to_latent = to_latent
        self.mlp_head = mlp_head
    
    def forward(self, x):
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ViT(nn.Module):
    def __init__(self, *, image_size=32, patch_size=4, num_classes=10, dim=512, depth=6, heads=8, mlp_dim=512, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        cls_token = nn.Parameter(torch.randn(1, 1, dim))
        dropout_layer = nn.Dropout(emb_dropout)

        self.pre_encoding = EncodingLayer(to_patch_embedding=to_patch_embedding,
                                          pos_embedding=pos_embedding,
                                          cls_token=cls_token,
                                          dropout=dropout_layer)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        pool = pool
        to_latent = nn.Identity()

        mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.classifier = Classifier(pool=pool, to_latent=to_latent, mlp_head=mlp_head)


    def forward(self, img):
        x = self.pre_encoding(img)

        x = self.transformer(x)

        x = self.classifier(x)
        return x

    def dump_layers(self) -> Tuple[List[nn.Module], List[nn.Module]]:
        """Dump each layer of the model

        Returns:
            Tuple[List[nn.Module], List[nn.Module]]: (feat_extractor, classifier),
                feat_extractor (List[nn.Module]) is the list of layers in the feature extractor
                classifier (List[nn.Module]) is the list of layers in the classifyer
        """
        return ([self.pre_encoding] + list(self.transformer.layers), 
                self.classifier)