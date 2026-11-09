
from __future__ import annotations
import cv2, re, time
from collections import OrderedDict
from pathlib import Path
from ultralytics import YOLO
from plate_ocr import PlateOCR
from plate_sr_model import PlateRestorer

# COCO vehicle classes: car, motorcycle, bus, truck
VEHICLE_CLASSES={2,3,5,7}

class LicensePlatePipeline:
    def __init__(self,cfg):
        self.cfg=cfg
        self.vehicle_model=YOLO(cfg["vehicle_weights"])
        self.plate_ocr=PlateOCR(cfg.get("char_weights",""),cfg["char_confidence"])
        self.restorer=PlateRestorer(cfg.get("sr_weights",""),device=cfg.get("device","auto"),
                                    scale=int(cfg.get("sr_scale",4)))
        self.last_seen=OrderedDict()
        self.event_count=0
    def _plate_roi(self,frame,box):
        x1,y1,x2,y2=box; h,w=frame.shape[:2]
        # Plate is commonly in the lower half of a vehicle box; crop a conservative lower region.
        vh=y2-y1; vw=x2-x1
        rx1=max(0,x1+int(vw*.12)); rx2=min(w,x2-int(vw*.12))
        ry1=max(0,y1+int(vh*.48)); ry2=min(h,y2-int(vh*.02))
        if rx2<=rx1 or ry2<=ry1: return frame[max(0,y1):y2,max(0,x1):x2].copy()
        return frame[ry1:ry2,rx1:rx2].copy()
    def process(self,frame,frame_idx):
        result=self.vehicle_model.track(frame,persist=True,tracker="bytetrack.yaml",
                    conf=float(self.cfg["vehicle_confidence"]),verbose=False)[0]
        out=[]
        if result.boxes is None: return out
        for i,b in enumerate(result.boxes):
            cls=int(b.cls[0].cpu().item())
            if cls not in VEHICLE_CLASSES: continue
            x1,y1,x2,y2=map(int,b.xyxy[0].cpu().numpy())
            tid=int(result.boxes.id[i].cpu().item()) if result.boxes.id is not None else frame_idx*100+i
            crop=self._plate_roi(frame,(x1,y1,x2,y2))
            restored=self.restorer.restore(crop)
            text=self.plate_ocr.recognize(restored)
            text=re.sub(r"[^A-Z0-9.-]","",text.upper())
            now=time.time()
            if text and now-self.last_seen.get((tid,text),0)>float(self.cfg["event_cooldown"]):
                self.last_seen[(tid,text)]=now; self.event_count+=1
                print(f"[PLATE] track={tid} text={text}")
            out.append({"vehicle_id":tid,"vehicle_box":(x1,y1,x2,y2),
                        "plate_crop":crop,"restored_crop":restored,"plate_text":text})
        return out
    def close(self): pass
