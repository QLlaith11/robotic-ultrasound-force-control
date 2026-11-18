import csv
from pathlib import Path
import numpy as np

def make_figure(metrics_csv="results/metrics.csv", output="results/figures/rmse.svg"):
    rows=list(csv.DictReader(Path(metrics_csv).open()))
    groups={}
    for r in rows:
        key=(r["organ"],r["controller"])
        groups.setdefault(key,[]).append(float(r["rmse"]))
    width,height=720,420
    vals=[v for x in groups.values() for v in x]
    vmax=max(vals+[1.]); sx=560/max(1,len(groups)); sy=320/vmax
    bars=[]
    for i,(key,values) in enumerate(sorted(groups.items())):
        mean=sum(values)/len(values); x=90+i*sx
        bars.append(f'<rect x="{x:.1f}" y="{360-mean*sy:.1f}" width="{sx*.65:.1f}" height="{mean*sy:.1f}" fill="#2878b5"/>')
        bars.append(f'<text x="{x:.1f}" y="385" font-size="11">{key[0]}-{key[1]}</text>')
        bars.append(f'<text x="{x:.1f}" y="{350-mean*sy:.1f}" font-size="10">{mean:.2f}</text>')
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/><text x="90" y="25" font-size="18">RMSE by organ/controller (N)</text>
<line x1="80" y1="360" x2="680" y2="360" stroke="black"/>{''.join(bars)}
<text x="15" y="200" transform="rotate(-90 15 200)" font-size="12">RMSE (N)</text></svg>'''
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(svg)
    return p

def make_metric_figures(metrics_csv="results/metrics.csv", output_dir="results/figures"):
    rows=list(csv.DictReader(Path(metrics_csv).open()))
    metrics=("rmse","mae","p95_abs_error","overshoot","settling_time","effort","pct_above_15")
    labels={"rmse":"RMSE (N)","mae":"MAE (N)","p95_abs_error":"95th percentile absolute error (N)",
            "overshoot":"Overshoot (N)","settling_time":"Settling time (s)",
            "effort":"Control effort","pct_above_15":"Time above 15 N (%)"}
    written=[]
    for metric in metrics:
        groups={}
        for row in rows:
            groups.setdefault((row["organ"],row["controller"]),[]).append(float(row[metric]))
        means=[(key,float(np.mean(values))) for key,values in sorted(groups.items())]
        vmax=max([value for _,value in means]+[1.])
        sx=560/max(1,len(means)); sy=320/vmax; bars=[]
        for i,(key,value) in enumerate(means):
            x=90+i*sx
            bars.append(f'<rect x="{x:.1f}" y="{360-value*sy:.1f}" width="{sx*.65:.1f}" height="{value*sy:.1f}" fill="#2878b5"/>')
            bars.append(f'<text x="{x:.1f}" y="385" font-size="11">{key[0]}-{key[1]}</text>')
            bars.append(f'<text x="{x:.1f}" y="{350-value*sy:.1f}" font-size="10">{value:.2f}</text>')
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420">
<rect width="100%" height="100%" fill="white"/><text x="90" y="25" font-size="18">{labels[metric]}</text>
<line x1="80" y1="360" x2="680" y2="360" stroke="black"/>{''.join(bars)}
<text x="15" y="200" transform="rotate(-90 15 200)" font-size="12">{labels[metric]}</text></svg>'''
        path=Path(output_dir)/f"{metric}.svg"; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(svg); written.append(path)
    return written
