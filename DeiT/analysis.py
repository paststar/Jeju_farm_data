

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os.path

import torch
import torch.nn as nn
import torch.utils.data as torchdata

from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import trunc_normal_
import glob
from datasets.dataset import get_dataset

class analysis_DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))

        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        #self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.head_dist.apply(self._init_weights)
        self.head.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        return x, x_dist

        # x = self.head(x)
        # x_dist = self.head_dist(x_dist)
        # if self.training:
        #     return x, x_dist
        # else:
        #     # during inference, return the average of both classifier predictions
        #     return (x + x_dist) / 2
        #     #return x_dist


class analysis_DomainVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trans_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.disc_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))

        self.head = nn.Linear(2 * self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_trans = nn.Linear(self.embed_dim, 2) if self.num_classes > 0 else nn.Identity()
        self.head_disc = nn.Linear(self.embed_dim, 2) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.trans_token, std=.02)
        trunc_normal_(self.disc_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.head.apply(self._init_weights)
        self.head_trans.apply(self._init_weights)
        self.head_disc.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        trans_token = self.trans_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        disc_token = self.disc_token.expand(B, -1, -1)

        x = torch.cat((trans_token, disc_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x_trans, x_disc = self.forward_features(x)
        x = torch.cat((x_trans, x_disc), dim=1)
        x = self.head(x)

        return x_trans, x_disc
        # if self.training:
        #     # x_trans = self.head_trans(x_trans)
        #     # x_disc = self.head_disc(x_disc)
        #
        # else:
        #     return x


if __name__ == "__main__":
    model = analysis_DomainVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=31).cuda()
    model.default_cfg = _cfg()
    #model_path = "./logs/pretrained_DeiT/add"
    model_path = "./logs/pretrained_DeiT/domain/CE"
    domain_dict = {"D" : "dslr", "A":"amazon", "W":"webcam"}

    token = {}
    embed = {}
    for path in glob.glob(model_path+"/*/*.pt"):
    #for path in glob.glob("./logs/pretrained_DeiT/domain/DeiT_S_domain _1228_1450_AtoW/*.pt"):
        source = domain_dict[path.split('/')[-2].split('_')[-1][-4]]
        target = domain_dict[path.split('/')[-2].split('_')[-1][-1]]

        tgt_trainset, tgt_testset = get_dataset(target, path="../Dataset")

        tgt_test_loader = torchdata.DataLoader(tgt_testset, batch_size=64, shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=False)
        model.load_state_dict(torch.load(os.path.join(path)))
        model.eval()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        res = torch.tensor([]).cuda()
        sum = 0
        token[source[0].upper()+"to"+target[0].upper()] =cos(model.trans_token[0],model.disc_token[0]).item()
        #print("token : ",)

        with torch.no_grad():
            for step, tgt_data in enumerate(tgt_test_loader):
                tgt_imgs, tgt_labels = tgt_data
                tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)
                embed_src ,embed_tgt  = model(tgt_imgs)
                res = torch.cat([res,cos(embed_src, embed_tgt)])
            embed[source[0].upper() + "to" + target[0].upper()] = (torch.sum(res)/len(res)).item()

            #print("embed : ", )
    print()
    print(model_path)
    print("############### cosine similarity ###############")
    key = " "*8
    value1 = ""
    value2 = ""
    #for i in ["AtoW", "DtoW", "WtoD", "AtoD", "DtoA", "WtoA"]:
    for i in ["AtoW"]:
        key += i + "\t"
        value1 += "{:.4f}\t".format(token[i])
        value2 += "{:.4f}\t".format(embed[i])

    print(key)
    print("token ", value1)
    print("embed ", value2)