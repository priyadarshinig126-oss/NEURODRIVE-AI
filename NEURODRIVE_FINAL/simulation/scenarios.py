import cv2,numpy as np
class DemoSimulation:
    def frames(self,n=120,width=960,height=540):
        for i in range(n):
            f=np.full((height,width,3),35,dtype=np.uint8)
            cv2.fillPoly(f,[np.array([(int(width*.28),height),(int(width*.43),int(height*.50)),(int(width*.57),int(height*.50)),(int(width*.72),height)],np.int32)],(65,65,65))
            for y in range(int(height*.55),height,45):cv2.line(f,(width//2-10,y),(width//2+10,y+20),(220,220,220),4)
            phase=(i%80)/80;oh=int(45+150*phase);cx=width//2+int(80*np.sin(i/18));cy=int(height*.53+oh*.75);ow=int(oh*1.65);x1,y1=cx-ow//2,cy-oh;x2,y2=cx+ow//2,cy
            cv2.rectangle(f,(x1,y1),(x2,y2),(50,80,190),-1);cv2.rectangle(f,(x1,y1),(x2,y2),(240,240,240),2);cv2.putText(f,"SIM VEHICLE",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            if 35<i%100<55:
                px=int(width*.68);py=int(height*.76);cv2.circle(f,(px,py-45),14,(220,180,120),-1);cv2.rectangle(f,(px-10,py-32),(px+10,py+10),(220,180,120),-1)
            yield f
