"""kp920: python3 test/run_l1_capacity.py
Probe B block + two A micro-panels, ir -> jr; one fixed result/error pair.
Input-size budget excludes the separately reported active C block and gaps.
"""
from pathlib import Path
import subprocess
import random
import sys
root=Path(__file__).resolve().parents[1]
binary=root/'test/l1_capacity'
with (root/'result.out').open('a' if '--refine' in sys.argv else 'w') as out,(root/'error.err').open('a' if '--refine' in sys.argv else 'w') as err:
    def emit(s):
        print(s,flush=True);out.write(s+'\n');out.flush()
    for name in ('l1d_cache_refill','l1d_cache','l2d_cache_refill'):
        emit(name+'='+Path('/sys/bus/event_source/devices/armv8_pmuv3_0/events',name).read_text().strip())
    emit('CPU=0 GCC='+subprocess.check_output(['g++','-dumpfullversion'],text=True).strip())
    emit('flags=-O3 -std=c++17 -march=armv8-a+simd; kernel=2unroll_pipeline_2buffer; repetitions=7')
    emit('Model: B=NC*KC floats, A=2*MR*KC floats; active C=2*MR*NC floats; ir -> jr')
    subprocess.run(['g++','-O3','-std=c++17','-march=armv8-a+simd',str(root/'test/l1_capacity.cpp'),str(root/'src/neon_4x16_2unroll_pipeline_2buffer.cpp'),'-o',str(binary)],stdout=out,stderr=err,check=True)
    cases=set()
    for nc in (32,64,128):
        for kib in (24,32,40,44,48,52,56,60,64,68,72,80):
            kc=round(kib*1024/(4*(nc+8))/4)*4
            for gap in (0,64,4096): cases.add((nc,kc,gap))
    if '--refine' in sys.argv:
        emit('REFINEMENT: 36..48 KiB, four shapes/four address gaps; plus regular candidates')
        cases=set()
        for nc in (16,32,64,128):
            for kib in (36,40,42,44,48):
                kc=int(kib*1024/(4*(nc+8))/4)*4
                for gap in (0,64,1024,4096): cases.add((nc,kc,gap))
        for nc,kc in ((16,384),(32,256),(64,128),(128,64)):
            for gap in (0,64,1024,4096): cases.add((nc,kc,gap))
    cases=sorted(cases);random.Random(1).shuffle(cases)
    for nc,kc,gap in cases:
        p=subprocess.run([str(binary),str(nc),str(kc),str(gap)],stdout=subprocess.PIPE,stderr=err,text=True)
        out.write(p.stdout);out.flush()
        if p.returncode: raise RuntimeError(f'case {nc,kc,gap} failed')
        for line in p.stdout.splitlines():
            if line.startswith('SUMMARY'): print(line,flush=True)
