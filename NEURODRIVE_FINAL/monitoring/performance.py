import statistics
class PerformanceMonitor:
    def __init__(self):self.values=[]
    def update(self,fps):self.values.append(float(fps));self.values=self.values[-300:]
    def summary(self):return {"avg_fps":round(statistics.mean(self.values),2) if self.values else 0,"min_fps":round(min(self.values),2) if self.values else 0,"samples":len(self.values)}
