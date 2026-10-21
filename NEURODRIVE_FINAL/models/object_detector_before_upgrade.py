from pathlib import Path
import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self,model_path):
        Path(model_path).parent.mkdir(parents=True,exist_ok=True)
        self.model=YOLO(model_path)
        self.allowed={"person","bicycle","car","motorcycle","bus","truck","traffic light","stop sign"}
        self.memory={}; self.next_id=1
    def detect(self,frame,conf=.35):
        results=self.model.predict(frame,conf=conf,verbose=False)
        out=[]
        if not results:return out
        names=results[0].names
        for b in results[0].boxes:
            name=names[int(b.cls[0])]
            if name not in self.allowed:continue
            xyxy=b.xyxy[0].cpu().numpy().tolist()
            out.append({"class_name":name,"confidence":float(b.conf[0]),"bbox":[int(x) for x in xyxy]})
        return out
    @staticmethod
    def iou(a,b):
        ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
        ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        aa=max(1,ax2-ax1)*max(1,ay2-ay1); ab=max(1,bx2-bx1)*max(1,by2-by1)
        return inter/(aa+ab-inter) if aa+ab-inter else 0
    def track_ids(self,dets):
        used=set()
        for d in dets:
            best=None; score=0
            for tid,o in self.memory.items():
                if tid in used or o["class_name"]!=d["class_name"]:continue
                s=self.iou(o["bbox"],d["bbox"])
                if s>score:score=s;best=tid
            if best is None or score<.20:best=self.next_id;self.next_id+=1
            d["track_id"]=best;used.add(best)
        self.memory={d["track_id"]:{"bbox":d["bbox"],"class_name":d["class_name"]} for d in dets}
        return dets
    def draw(self,frame,dets):
        for d in dets:
            x1,y1,x2,y2=d["bbox"]; label=f'#{d["track_id"]} {d["class_name"]} {d["confidence"]:.0%}'
            if d.get("distance_m") is not None:label+=f' | {d["distance_m"]:.1f}m'
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,180,0),2)
            cv2.putText(frame,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),2)
            cv2.putText(frame,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1)
        return frame
