from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
def build_pdf_report(events,performance,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True);doc=SimpleDocTemplate(path,pagesize=A4);styles=getSampleStyleSheet();story=[Paragraph('NEURODRIVE-AI Safety & Autonomy Report',styles['Title']),Spacer(1,12),Paragraph(f'Average FPS: {performance.get("avg_fps",0)} | Minimum FPS: {performance.get("min_fps",0)} | Samples: {performance.get("samples",0)}',styles['BodyText']),Spacer(1,14)]
    rows=[["Frame","Action","Risk","Score","Distance","TTC","Driver","FPS"]]
    for e in events[-100:]:rows.append([e.get("frame"),e.get("action"),e.get("risk"),e.get("risk_score"),e.get("min_distance_m"),e.get("ttc_s"),e.get("driver_state"),e.get("fps")])
    if len(rows)==1:rows.append(['-','No events','-','-','-','-','-','-'])
    t=Table(rows,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1f2937')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),7)]));story.append(t);doc.build(story);return path
