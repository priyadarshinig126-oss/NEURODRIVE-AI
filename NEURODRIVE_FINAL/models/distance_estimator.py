class DistanceEstimator:
    HEIGHTS = {"person":1.70,"bicycle":1.50,"car":1.50,"motorcycle":1.25,"bus":3.20,"truck":3.20,"stop sign":0.75,"traffic light":0.70}
    def estimate(self,bbox,frame_shape,reference_height_m=1.5,focal_px=850.0,class_name="car"):
        x1,y1,x2,y2=bbox
        pixel_height=max(2.0,float(y2-y1))
        real_height=self.HEIGHTS.get(class_name,reference_height_m)
        return max(0.5,min(focal_px*real_height/pixel_height,200.0))
