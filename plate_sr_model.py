
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,nfeat):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(nfeat,nfeat,3,1,1),nn.ReLU(True),
                                 nn.Conv2d(nfeat,nfeat,3,1,1))
    def forward(self,x): return x+self.body(x)*0.2

class PlateRestorationNet(nn.Module):
    """Compact residual super-resolution network, compatible with this project's own weights."""
    def __init__(self,nin=3,nfeat=64,gc=32,nblock=5,scale=4):
        super().__init__()
        self.scale=scale
        layers=[nn.Conv2d(nin,nfeat,3,1,1),nn.ReLU(True)]
        layers += [ResidualBlock(nfeat) for _ in range(nblock)]
        layers += [nn.Conv2d(nfeat,nfeat,3,1,1),nn.PixelShuffle(1)]
        self.body=nn.Sequential(*layers)
        self.up=nn.Sequential(nn.Conv2d(nfeat,nfeat*scale*scale,3,1,1),
                              nn.PixelShuffle(scale),nn.Conv2d(nfeat,3,3,1,1))
    def forward(self,x): return self.up(self.body(x))

class PlateRestorer:
    def __init__(self,weights_path="",device="auto",lr_size=(32,16),nfeat=64,gc=32,nblock=5,scale=4):
        self.device=torch.device("cuda" if device=="auto" and torch.cuda.is_available()
                                 else ("cpu" if device=="auto" else device))
        self.lr_size=tuple(lr_size); self.scale=scale; self.model=None
        if weights_path and Path(weights_path).exists():
            self.model=PlateRestorationNet(3,nfeat,gc,nblock,scale).to(self.device)
            ckpt=torch.load(weights_path,map_location=self.device)
            state=ckpt.get("model",ckpt) if isinstance(ckpt,dict) else ckpt
            self.model.load_state_dict(state,strict=False); self.model.eval()
            print(f"Loaded plate SR weights: {weights_path}")
        else:
            print("SR weights unavailable: using high-quality bicubic/CLAHE fallback.")
    def restore(self,plate_bgr):
        if plate_bgr is None or plate_bgr.size==0: return plate_bgr
        if self.model is None:
            h,w=plate_bgr.shape[:2]
            up=cv2.resize(plate_bgr,(max(1,w*self.scale),max(1,h*self.scale)),interpolation=cv2.INTER_CUBIC)
            lab=cv2.cvtColor(up,cv2.COLOR_BGR2LAB)
            l,a,b=cv2.split(lab); l=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
            return cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
        rgb=cv2.cvtColor(plate_bgr,cv2.COLOR_BGR2RGB)
        t=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().to(self.device)/255.
        with torch.no_grad(): out=self.model(t).clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
        return cv2.cvtColor((out*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
