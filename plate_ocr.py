
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# Class-index map visible in the recording. It maps character detector IDs to glyphs.
CHAR_MAP={
 0:"O",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
 10:"A",11:"B",12:"C",13:"D",14:"E",15:"F",16:"G",17:"H",
 18:"K",19:"L",20:"M",21:"N",22:"P",23:"Q",24:"R",25:"S",
 26:"T",27:"U",28:"V",29:"X",30:"Y",31:"Z"
}

class PlateOCR:
    """YOLO character detector; sorts detections into one or two text rows."""
    def __init__(self,model_path="",conf=0.3):
        self.conf=float(conf)
        self.model=YOLO(model_path) if model_path and Path(model_path).exists() else None
        if self.model is None:
            print("Character weights not found: OCR disabled until --char-weights is supplied.")
    def recognize(self,plate_img):
        if plate_img is None or plate_img.size==0 or self.model is None: return ""
        h,w=plate_img.shape[:2]
        if w<1 or h<5: return ""
        results=self.model.predict(plate_img,conf=self.conf,verbose=False)
        if not results or results[0].boxes is None: return ""
        detections=[]
        for box in results[0].boxes:
            x1,y1,x2,y2=box.xyxy[0].detach().cpu().numpy()
            cls_id=int(box.cls[0].item())
            cx,cy=(x1+x2)/2,(y1+y2)/2
            char=CHAR_MAP.get(cls_id,"?")
            if char!="?": detections.append((cx,cy,char))
        if not detections: return ""
        ys=[d[1] for d in detections]
        y_range=max(ys)-min(ys)
        if y_range>h*.30 and len(detections)>3:
            y_mid=(min(ys)+max(ys))/2
            line1=sorted([d for d in detections if d[1]<y_mid],key=lambda z:z[0])
            line2=sorted([d for d in detections if d[1]>=y_mid],key=lambda z:z[0])
            chars=line1+line2
        else:
            chars=sorted(detections,key=lambda z:z[0])
        return "".join(d[2] for d in chars)
