import cv2
import numpy as np
import mediapipe as mp
class DriverMonitor:
    def __init__(self):
        self.mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5)
        self.closed=0;self.yawn=0
    @staticmethod
    def dist(a,b):return np.linalg.norm(np.array(a)-np.array(b))
    def ear(self,p,ids):
        q=[p[i] for i in ids];return (self.dist(q[1],q[5])+self.dist(q[2],q[4]))/(2*self.dist(q[0],q[3])+1e-6)
    def analyze(self,frame):
        h,w=frame.shape[:2];res=self.mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:return {"state":"NO_FACE","eye_score":0,"yawn_score":0,"attention":"UNKNOWN","seatbelt":"UNKNOWN","confidence":0}
        pts=[(int(p.x*w),int(p.y*h)) for p in res.multi_face_landmarks[0].landmark]
        ear=(self.ear(pts,[33,160,158,133,153,144])+self.ear(pts,[362,385,387,263,373,380]))/2
        mouth=self.dist(pts[13],pts[14])/(self.dist(pts[61],pts[291])+1e-6)
        self.closed=self.closed+1 if ear<.20 else max(0,self.closed-2); self.yawn=self.yawn+1 if mouth>.45 else max(0,self.yawn-2)
        state="DROWSY / EYES CLOSED" if self.closed>=12 else "YAWNING" if self.yawn>=10 else "AWAKE"
        nx=pts[1][0]/max(1,w);attention="FORWARD" if .38<=nx<=.62 else "LOOKING LEFT" if nx<.38 else "LOOKING RIGHT"
        return {"state":state,"eye_score":round(float(ear),3),"yawn_score":round(float(mouth),3),"attention":attention,"seatbelt":self.seatbelt(frame),"confidence":.82}
    def seatbelt(self,frame):
        h,w=frame.shape[:2];crop=frame[int(h*.48):];
        if crop.size==0:return "UNKNOWN"
        edges=cv2.Canny(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),60,140);lines=cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength=35,maxLineGap=15);n=0
        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0]:
                dx,dy=x2-x1,y2-y1
                if abs(dx)>10 and abs(dy)>10 and .35<abs(dy/dx)<3:n+=1
        return "LIKELY FASTENED" if n>=2 else "CHECK SEATBELT"
    def draw(self,frame,info):
        text=f"DRIVER: {info['state']} | ATTENTION: {info['attention']} | BELT: {info['seatbelt']}"
        cv2.rectangle(frame,(10,10),(min(frame.shape[1]-10,760),48),(20,20,20),-1);cv2.putText(frame,text,(20,37),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2);return frame
