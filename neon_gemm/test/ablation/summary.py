from pathlib import Path
from statistics import median
import re
rows={}
for line in Path('result.out').read_text().splitlines():
    m=re.match(r'RESULT round=(\d+) variant=(\S+) median_ms=([\d.]+)',line)
    if m: rows.setdefault(m[2],[]).append(float(m[3]))
assert len(rows)==11 and all(len(t)==3 for t in rows.values())
base=median(rows['V0_register']); final=median(rows['V4_vector_pack'])
print('SUMMARY: median of 3 round medians; speedup relative to V0; time increase relative to V4')
for name,t in rows.items():
    ms=median(t)
    print(f'{name:24s} ms={ms:.6f} GFLOPS={2*1024**3/ms/1e6:.6f} speedup={base/ms:.4f} time_vs_V4={(ms/final-1)*100:.3f}% round_medians={t}')
