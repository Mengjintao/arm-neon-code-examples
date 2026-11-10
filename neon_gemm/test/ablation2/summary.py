from pathlib import Path
from statistics import median
import re, sys

rows = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    m = re.match(r'RESULT round=(\d+) variant=(\S+) median_ms=([\d.]+)', line)
    if m: rows.setdefault(m[2], []).append(float(m[3]))
assert len(rows) == 15 and all(len(t) == 3 for t in rows.values()), {k: len(v) for k, v in rows.items()}
g = {k: 2 * 1024 ** 3 / median(v) / 1e6 for k, v in rows.items()}
ms = {k: median(v) for k, v in rows.items()}

ladder = ['S0_register_128', 'S1_unroll2_128', 'S2_pipeline_128', 'S3_pipeline_blocks',
          'V2_pipeline', 'V3_prefetch', 'V4_vector_pack']
print('SUMMARY ladder: median of 3 round medians; change = GFLOPS vs previous row')
prev = None
for k in ladder:
    ch = '' if prev is None else f'{(g[k] / g[prev] - 1) * 100:+.1f}%'
    print(f'LADDER {k:22s} ms={ms[k]:.3f} GFLOPS={g[k]:.3f} change={ch} rounds={rows[k]}')
    prev = k
print('SUMMARY removals from V4: slower = time vs V4 - 1')
for k in ['drop_A_prefetch', 'drop_B_prefetch', 'drop_C_prefetch', 'drop_all_prefetch', 'drop_pipeline']:
    print(f'DROP {k:22s} ms={ms[k]:.3f} GFLOPS={g[k]:.3f} slower={(ms[k] / ms["V4_vector_pack"] - 1) * 100:.1f}% rounds={rows[k]}')
print('OTHER')
for k in ['V0_register', 'V1_unroll2', 'plain_unroll4']:
    print(f'OTHER {k:22s} ms={ms[k]:.3f} GFLOPS={g[k]:.3f} rounds={rows[k]}')
