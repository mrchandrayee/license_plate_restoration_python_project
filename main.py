
import argparse, json, time
from pathlib import Path
import cv2
from pipeline import LicensePlatePipeline
from hud import HUDPanel, draw_corner_bbox

def main():
    ap=argparse.ArgumentParser(description="License Plate Restoration Video Pipeline")
    ap.add_argument("--source",default="0",help="Camera index or input video")
    ap.add_argument("--vehicle-weights",default=None)
    ap.add_argument("--char-weights",default=None)
    ap.add_argument("--sr-weights",default=None)
    ap.add_argument("--output",default=None)
    ap.add_argument("--no-display",action="store_true")
    args=ap.parse_args()
    cfg=json.load(open("config/settings.json",encoding="utf-8"))
    for key,val in [("vehicle_weights",args.vehicle_weights),("char_weights",args.char_weights),("sr_weights",args.sr_weights),("output",args.output)]:
        if val: cfg[key]=val
    source=int(args.source) if args.source.isdigit() else args.source
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f"Cannot open source: {source}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps<1 or fps>120: fps=25.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    out=Path(cfg["output"]); out.parent.mkdir(parents=True,exist_ok=True)
    panel=HUDPanel(width=cfg["panel_width"],max_entries=cfg["panel_entries"])
    pipeline=LicensePlatePipeline(cfg)
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps,
                           (width+cfg["panel_width"]+cfg["panel_margin"],height))
    frame_idx=0; t0=time.time(); fps_now=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame_idx+=1
            detections=pipeline.process(frame,frame_idx)
            canvas=frame.copy()
            for det in detections:
                x1,y1,x2,y2=det["vehicle_box"]
                draw_corner_bbox(canvas,(x1,y1),(x2,y2),(60,230,100),3,7,28)
                label=f"VEHICLE {det['vehicle_id']} | {det['plate_text'] or 'PLATE NOT READ'}"
                cv2.putText(canvas,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,(80,255,120),1,cv2.LINE_AA)
                panel.update(det["vehicle_id"],det["plate_crop"],det["restored_crop"],det["plate_text"],frame_idx)
            if frame_idx%25==0: fps_now=frame_idx/max(time.time()-t0,1e-6)
            combined=panel.render(canvas,fps_now,frame_idx)
            writer.write(combined)
            if not args.no_display:
                cv2.imshow("License Plate Restoration",combined)
                if cv2.waitKey(1)&0xff in (ord("q"),27): break
    finally:
        cap.release(); writer.release(); cv2.destroyAllWindows()
        pipeline.close()
    print(f"Saved: {out.resolve()}")
    print(f"Recognized plate events: {pipeline.event_count}")

if __name__=="__main__": main()
