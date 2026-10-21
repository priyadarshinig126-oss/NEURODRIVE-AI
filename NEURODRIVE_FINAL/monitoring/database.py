import sqlite3
from pathlib import Path
from datetime import datetime
class EventLogger:
    def __init__(self,path="logs/neurodrive.db"):
        Path(path).parent.mkdir(parents=True,exist_ok=True);self.conn=sqlite3.connect(path,check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT,frame INTEGER,action TEXT,risk TEXT,risk_score REAL,min_distance REAL,ttc REAL,driver_state TEXT,fps REAL)");self.conn.commit()
    def log(self,e):
        self.conn.execute("INSERT INTO events(timestamp,frame,action,risk,risk_score,min_distance,ttc,driver_state,fps) VALUES(?,?,?,?,?,?,?,?,?)",(datetime.now().isoformat(timespec="seconds"),e["frame"],e["action"],e["risk"],e["risk_score"],e["min_distance_m"],e["ttc_s"],e["driver_state"],e["fps"]));self.conn.commit()
