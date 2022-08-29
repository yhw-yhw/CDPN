import os
from easydict import EasyDict as edict

import torch
import torch.nn as nn

class CDPN(nn.Module):
    def __init__(self, backbone, rot_head_net, trans_head_net):
        super(CDPN, self).__init__()
        self.backbone = backbone
        self.rot_head_net = rot_head_net
        self.trans_head_net = trans_head_net

        self.linear=nn.Linear(64*4, 3)
        self.linear2=nn.Linear(64*64*4, 8*3)
        self.relu=nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x.shape (8, 240, 180, 64, 4)
        x=torch.flatten(x,start_dim=3)
        # x.shape (8, 240, 180, 64*4)
        x=x.view(8*240*180, 64*4)
        # x.shape (8*240*180, 64*4)
        x=self.linear(x)
        x=self.relu(x)
        # x.shape (8*240*180, 3)
        x=x.view(8, 240, 180, 3)
        # x.shape (8, 240, 180, 3)
        x=x.permute(0,3,1,2).contiguous()
        # x.shape (8, 3, 240, 180)
        x=torch.nn.functional.pad(x,(38,38,8,8))
        # torch.Size([8, 3, 256, 256])
            
        features = self.backbone(x)           
        # torch.Size([8, 512, 8, 8])
        
        freeze_rot=0
        if freeze_rot:
            with torch.no_grad():
                cc_maps = self.rot_head_net(features) 
        else:
            cc_maps = self.rot_head_net(features) 
        # torch.Size([8, 4, 64, 64])
        
        trans = self.trans_head_net(features)
        # torch.Size([8, 3])
        
        cc_maps=torch.flatten(cc_maps,start_dim=1)
        cc_maps=self.linear2(cc_maps)
        # cc_maps=self.relu(cc_maps)
        cc_maps=cc_maps.view(8,1,8,3)
        # --> (8, 1, 8, 3)
        return cc_maps, trans
