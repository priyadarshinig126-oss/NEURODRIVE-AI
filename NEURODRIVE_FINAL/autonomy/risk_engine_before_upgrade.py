class RiskEngine:
    def assess(self,objects,lane):
        closest=None;dmin=999.
        for d in objects:
            if d["class_name"] in {"car","truck","bus","motorcycle","person","bicycle"} and d.get("distance_m",999)<dmin:dmin=d["distance_m"];closest=d
        proximity=max(0,min(100,(20-dmin)*5));lane_risk=min(25,abs(lane["offset"])*50);score=min(100,proximity+lane_risk)
        level="CRITICAL" if dmin<5 else "HIGH" if dmin<10 else "MEDIUM" if dmin<18 else "LOW"
        ttc=closest.get("ttc_s") if closest else None
        if ttc is not None and ttc<2:level="CRITICAL";score=max(score,90)
        elif ttc is not None and ttc<4:level="HIGH";score=max(score,70)
        return {"risk_level":level,"risk_score":round(score,1),"min_distance_m":round(dmin,1) if closest else None,"ttc_s":ttc,"closest":closest}
