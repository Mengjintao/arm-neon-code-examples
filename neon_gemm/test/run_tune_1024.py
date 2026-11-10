#!/usr/bin/env python3
"""Run on kp920 from gemm_neon. Append stages to the same result.out/error.err.
First compile test/tune_1024.cpp with src/pack_v.cpp,
src/neon_4x16_2unroll_pipeline_2buffer.cpp and src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp.
Run `test/tune_1024 micro >result.out 2>error.err` before this script.
"""
import itertools
import subprocess
import sys

phase = sys.argv[1]
with open('result.out', 'a', buffering=1) as out, open('error.err', 'a') as err:
    def run(args, data=None):
        subprocess.run(['./test/tune_1024', *map(str,args)], input=data, text=True,
                       stdout=out, stderr=err, check=True)
    if phase == 'explore':
        out.write('STAGE packed_reuse\n')
        for k in (256,384,512,768,1024):
            for m in (16,32,64,128,256):
                run(['probe',m,16,k])
            for m,n in itertools.product((32,64,128),(64,128,256,512)):
                run(['probe',m,n,k])
        out.write('STAGE full_gemm_screen N=1024 rounds=3\n')
        candidates=list(itertools.product((32,64,96,128,256),(64,128,160,256,384,512,1024),(128,256,384,512,768,1024)))
        run([3], ''.join(f'{m} {n} {k}\n' for m,n,k in candidates))
    elif phase == 'refine':
        out.write('STAGE full_gemm_refine N=1024 rounds=7\n')
        c=set(itertools.product((40,48,56,64,72,80),(1024,),(256,320,344,352,384,448,512)))
        c.update(itertools.product((16,24,32,48),(256,384,512),(256,384)))
        c.update((64,n,384) for n in (640,768,896))
        c.update(((64,160,256),(96,128,256),(128,256,256),(64,1024,576),(64,1024,640)))
        run([7], ''.join(f'{m} {n} {k}\n' for m,n,k in sorted(c)))
    elif phase == 'local':
        out.write('STAGE local_refine N=1024 rounds=5\n')
        c=set(itertools.product((8,12,16,20,24,28,32),(192,224,256,288,320,352,384),(256,)))
        c.update(itertools.product((16,24,32),(256,),(128,192,224,288,320,384)))
        c.update(((64,1024,512),(64,1024,384),(64,160,256),(96,128,256)))
        run([5], ''.join(f'{m} {n} {k}\n' for m,n,k in sorted(c)))
    else:
        raise SystemExit('phase must be explore, refine or local')
