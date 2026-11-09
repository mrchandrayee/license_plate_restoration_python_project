
from collections import OrderedDict
import cv2, numpy as np
def draw_corner_bbox(img,pt1,pt2,color,thickness=3,corner_thick=6,corner_len=35):
    x1,y1=pt1; x2,y2=pt2
    cv2.rectangle(img,(x1,y1),(x2,y2),color,thickness)
    length=min(corner_len,max(12,min(x2-x1,y2-y1)//3))
    corners=[(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]
    for x,y,dx,dy in corners:
        cv2.line(img,(x,y),(x+dx*length,y),color,corner_thick)
        cv2.line(img,(x,y),(x,y+dy*length),color,corner_thick)

def draw_bold_label(img,text,pos,font_scale=.8,font_thick=2,text_color=(255,255,255),
                    bg_color=(15,20,25),border_color=None,pad=6):
    x,y=pos; (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,font_scale,font_thick)
    x1=max(0,x); y1=max(0,y-th-pad*2); x2=min(img.shape[1]-1,x1+tw+pad*2); y2=min(img.shape[0]-1,y+pad)
    cv2.rectangle(img,(x1,y1),(x2,y2),bg_color,-1)
    if border_color: cv2.rectangle(img,(x1,y1),(x2,y2),border_color,1)
    cv2.putText(img,text,(x1+pad,y1+th+pad//2),cv2.FONT_HERSHEY_SIMPLEX,font_scale,text_color,font_thick,cv2.LINE_AA)

class HUDPanel:
    def __init__(self,width=580,margin_right=10,max_entries=1):
        self.width=width; self.margin_right=margin_right; self.max_entries=max_entries
        self.entries=OrderedDict()
    def update(self,track_id,crop_img,restored_img,ocr_text,frame_idx):
        self.entries[track_id]={"crop":crop_img.copy() if crop_img is not None else None,
            "restored":restored_img.copy() if restored_img is not None else None,
            "text":ocr_text,"time":frame_idx}
        self.entries.move_to_end(track_id)
        while len(self.entries)>self.max_entries: self.entries.popitem(last=False)
    def render(self,frame,fps=0,frame_idx=0):
        fh,fw=frame.shape[:2]; panel_w=self.width
        canvas=np.zeros((fh,fw+panel_w+self.margin_right,3),dtype=np.uint8)
        canvas[:,:fw]=frame
        px=fw+self.margin_right
        canvas[:,px:px+panel_w]=(14,16,24)
        cv2.putText(canvas,"LICENSE PLATE RESTORATION",(px+12,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(70,210,255),1,cv2.LINE_AA)
        cv2.putText(canvas,f"FPS {fps:.1f} | FRAME {frame_idx}",(px+12,54),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1,cv2.LINE_AA)
        if not self.entries:
            cv2.putText(canvas,"Waiting for vehicle...",(px+14,95),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,180,180),1)
            return canvas
        _,entry=list(self.entries.items())[-1]
        y=78
        for title,key in (("PLATE CROP","crop"),("RESTORED PLATE","restored")):
            cv2.putText(canvas,title,(px+12,y+16),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,220,180),1,cv2.LINE_AA)
            y+=24
            im=entry[key]
            if im is not None:
                maxw=panel_w-24; maxh=max(65,int(fh*.23))
                scale=min(maxw/im.shape[1],maxh/im.shape[0])
                resized=cv2.resize(im,(max(1,int(im.shape[1]*scale)),max(1,int(im.shape[0]*scale))))
                canvas[y:y+resized.shape[0],px+12:px+12+resized.shape[1]]=resized
                y+=maxh+18
        cv2.putText(canvas,"OCR RESULT",(px+12,min(fh-48,y+12)),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,220,180),1)
        cv2.putText(canvas,entry["text"] or "NOT RECOGNIZED",(px+12,min(fh-18,y+38)),
                    cv2.FONT_HERSHEY_SIMPLEX,.62,(255,220,80),2,cv2.LINE_AA)
        return canvas
