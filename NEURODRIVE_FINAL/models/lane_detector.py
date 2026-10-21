import cv2
import numpy as np
class LaneDetector:
    def detect(self,frame):
        h,w=frame.shape[:2]; roi=frame[int(h*.55):]
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY); edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),50,150)
        mask=np.zeros_like(edges); poly=np.array([[(int(w*.05),roi.shape[0]),(int(w*.42),int(roi.shape[0]*.05)),(int(w*.58),int(roi.shape[0]*.05)),(int(w*.95),roi.shape[0])]],np.int32)
        cv2.fillPoly(mask,poly,255); edges=cv2.bitwise_and(edges,mask)
        lines=cv2.HoughLinesP(edges,1,np.pi/180,35,minLineLength=35,maxLineGap=80); left=[];right=[]
        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0]:
                slope=(y2-y1)/((x2-x1) or 1)
                if abs(slope)<.35:continue
                (left if slope<0 else right).append((x1,y1,x2,y2))
        l=self.avg(left,roi.shape[0]); r=self.avg(right,roi.shape[0]); center=w/2
        if l and r: lane_center=(l[2]+r[2])/2; offset=(center-lane_center)/(w/2); confidence=.9
        else: lane_center=center;offset=0.;confidence=.25
        return {"left":l,"right":r,"lane_center":lane_center,"offset":float(offset),"confidence":confidence}
    def avg(self,lines,roi_h):
        if not lines:return None
        ms=[];bs=[]
        for x1,y1,x2,y2 in lines:
            m=(y2-y1)/((x2-x1) or 1);ms.append(m);bs.append(y1-m*x1)
        m=float(np.mean(ms));b=float(np.mean(bs));yb=roi_h;yt=int(roi_h*.15)
        return (int((yt-b)/m),yt,int((yb-b)/m),yb)
    def draw(self,frame,info):
        h,_=frame.shape[:2]; overlay=frame.copy()
        for side in ("left","right"):
            line=info.get(side)
            if line:
                x1,y1,x2,y2=line;cv2.line(overlay,(x1,int(h*.55)+y1),(x2,int(h*.55)+y2),(0,255,0),5)
        cv2.addWeighted(overlay,.65,frame,.35,0,frame);return frame
