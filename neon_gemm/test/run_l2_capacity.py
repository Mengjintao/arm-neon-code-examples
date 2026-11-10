"""Run on kp920: python3 test/run_l2_capacity.py
Steady-state packed A+B capacity sweep with real C loads/stores, jr -> ir.
Input budget and active C are reported separately. No packing in timed region.
"""
from pathlib import Path
import subprocess,random,sys
root=Path(__file__).resolve().parents[1]
binary=root/'test/l2_capacity'
with (root/'result.out').open('a' if '--refine' in sys.argv else 'w') as out,(root/'error.err').open('a' if '--refine' in sys.argv else 'w') as err:
    def emit(s):
        print(s,flush=True);out.write(s+'\n');out.flush()
    for name in ('l1d_cache_refill','l2d_cache','l2d_cache_refill'):
        emit(name+'='+Path('/sys/bus/event_source/devices/armv8_pmuv3_0/events',name).read_text().strip())
    emit('CPU=0 GCC='+subprocess.check_output(['g++','-dumpfullversion'],text=True).strip())
    emit('flags=-O3 -std=c++17 -march=armv8-a+simd; kernel=2unroll_pipeline_2buffer; repetitions=7')
    emit('Model: inputs=4*KC*(MC+NC); active C=4*MC*NC; jr -> ir; warm packed panels, no pack interference')
    subprocess.run(['g++','-O3','-std=c++17','-Wall','-Wextra','-Wpedantic','-march=armv8-a+simd',str(root/'test/l2_capacity.cpp'),str(root/'src/neon_4x16_2unroll_pipeline_2buffer.cpp'),'-o',str(binary)],stdout=out,stderr=err,check=True)
    cases=set()
    for kc,nc in ((256,128),(256,256),(128,256),(384,128)):
        for kib in (192,256,320,352,384,416,448,480,512,576,640):
            mc=int((kib*1024/(4*kc)-nc)/4)*4
            if mc<=0:continue
            for gap in (0,64,4096):cases.add((mc,nc,kc,gap,0))
    if '--refine' in sys.argv:
        emit('REFINEMENT: reduced inputs and scratch-C control')
        cases=set()
        for kc,nc in ((256,64),(256,128),(128,256),(256,256)):
            for kib in (128,160,192,224,256,288,320,384,448,512):
                mc=int((kib*1024/(4*kc)-nc)/4)*4
                if mc<=0:continue
                for scratch in (0,1):
                    if not scratch and kib>288:continue
                    for gap in (0,64,4096):cases.add((mc,nc,kc,gap,scratch))
    cases=sorted(cases);random.Random(2).shuffle(cases)
    for mc,nc,kc,gap,scratch in cases:
        p=subprocess.run([str(binary),str(mc),str(nc),str(kc),str(gap),str(scratch)],stdout=subprocess.PIPE,stderr=err,text=True)
        out.write(p.stdout);out.flush()
        if p.returncode:raise RuntimeError(f'case {mc,nc,kc,gap} failed')
        for line in p.stdout.splitlines():
            if line.startswith('SUMMARY'):print(line,flush=True)
