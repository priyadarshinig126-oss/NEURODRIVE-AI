import streamlit as st
import pandas as pd
def render_css():st.markdown("<style>.block-container{padding-top:1.2rem}.main{background:#070b12}.neuro-title{font-size:2.3rem;font-weight:800}.neuro-sub{color:#8fa3bf}</style>",unsafe_allow_html=True)
def render_header():st.markdown('<div class="neuro-title">🚗 NEURODRIVE-AI</div>',unsafe_allow_html=True);st.markdown('<div class="neuro-sub">Perceive → Track → Estimate → Predict → Assess Risk → Decide → Act → Explain</div>',unsafe_allow_html=True)
def render_metrics(ps,d,r,l,fps):
    for p,(lab,val) in zip(ps,[("Autonomy",d["action"]),("Risk",f'{r["risk_level"]} ({r["risk_score"]:.0f})'),("Distance","—" if r["min_distance_m"] is None else f'{r["min_distance_m"]:.1f} m'),("FPS",f'{fps:.1f}')]):p.metric(lab,val)
def render_decision(box,d,r):
    with box.container():st.markdown(f'### `{d["action"]}`');st.write(d["reason"]);st.progress(min(1,d["confidence"]),text=f'Decision confidence: {d["confidence"]:.0%}');st.caption(f'Estimated TTC: {r["ttc_s"]:.2f}s' if r["ttc_s"] is not None else 'TTC: not available')
def render_driver_card(box,x):
    with box.container():st.markdown(f'**State:** `{x["state"]}`');st.write(f'**Attention:** {x["attention"]}');st.write(f'**Seatbelt:** {x["seatbelt"]}');st.write(f'**Eye openness:** {x["eye_score"]}');st.write(f'**Yawn score:** {x["yawn_score"]}')
def render_event_table(events):
    if events:st.dataframe(pd.DataFrame(events),use_container_width=True,hide_index=True)
    else:st.info('No safety events yet.')
