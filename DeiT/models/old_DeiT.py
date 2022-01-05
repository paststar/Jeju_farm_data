import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from .DANN import ReverseLayerF

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224, domain_token: bool = True):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size), # original implemntation 이렇게 함
            Rearrange('b e (h) (w) -> b (h w) e'),
            # nn.Linear(patch_size * patch_size * in_channels, emb_size) # 논문대로 하면 이렇게 해야됨
        )
        self.src_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.tgt_token = nn.Parameter(torch.randn(1, 1, emb_size))

        if domain_token:
            self.domain_token = nn.Parameter(torch.randn(1, 1, emb_size))
            self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 3, emb_size))
        else:
            self.domain_token = None
            self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 2, emb_size))


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        src_token = repeat(self.src_token, '() n e -> b n e', b=b)
        tgt_token = repeat(self.tgt_token, '() n e -> b n e', b=b)

        if self.domain_token is not None:
            domain_token = repeat(self.domain_token, '() n e -> b n e', b=b)
            x = torch.cat([src_token, tgt_token, domain_token, x], dim=1)
        else:
            x = torch.cat([src_token, tgt_token, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0, mask: bool = True, num_token : int = 1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

        if mask:
            self.mask = torch.ones(num_token, num_token, dtype=torch.bool).cuda()
            self.mask[0][1] = 0
            self.mask[1][0] = 0
        else:
            self.mask = None

    #def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
    def forward(self, x: Tensor) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if self.mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~self.mask, fill_value) # masked_fill(mask(bool), value) : True 부분

        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 num_heads: int = 3,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size,num_heads, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class Classifier(nn.Module):
    alpha = 0
    def __init__(self, emb_size: int = 768, n_classes: int = 1000, n_domin: int = 2):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.n_domin = n_domin

        self.src_classifier = nn.Linear(emb_size, n_classes)
        self.tgt_classifier = nn.Linear(emb_size, n_classes)

        if self.n_domin != -1:
            self.domin_classifier = nn.Linear(emb_size, n_classes)
        else:
            self.domin_classifier = None

    def forward(self, x, **kwargs):
        if self.domin_classifier is not None:
            reversed_input = ReverseLayerF.apply(x[:, 2],Classifier.alpha)
            return self.src_classifier(x[:, 0]), self.tgt_classifier(x[:, 1]), self.domin_classifier(reversed_input)
        else:
            return self.src_classifier(x[:, 0]), self.tgt_classifier(x[:, 1])

    @staticmethod
    def set_alpha(alpha):
        Classifier.alpha = alpha

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

# class DeiT(nn.Sequential):
#     def __init__(self,
#                  in_channels: int = 3,
#                  patch_size: int = 16,
#                  emb_size: int = 768,
#                  num_heads: int = 12,
#                  img_size: int = 224,
#                  depth: int = 12,
#                  n_classes: int = 31,
#                  **kwargs):
#         super().__init__(
#             PatchEmbedding(in_channels, patch_size, emb_size, img_size),
#             TransformerEncoder(depth, emb_size=emb_size,num_heads=num_heads, **kwargs),
#             Classifier(emb_size, n_classes)
#         )

class DeiT_tiny(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 192,
                 num_heads: int = 3,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 31,
                 n_domain: int = -1,
                 mask: bool = True,
                 **kwargs):
        num_token = (img_size//patch_size)**2 + 2 + int(n_domain>0)
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size, domain_token=n_domain>0),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads, mask = mask, num_token = num_token, **kwargs),
            Classifier(emb_size, n_classes, n_domain)
        )

if __name__ == "__main__":
    summary(DeiT_tiny(), (3, 224, 224), device='cuda')

    #summary(DeiT_tiny(), (3, 224, 224), device='cpu')

    #test=DeiT()
    #test(torch.zeros((16,3, 224, 224)))
    #summary(ViT_tiny(), (3, 224, 224), device='cpu')

