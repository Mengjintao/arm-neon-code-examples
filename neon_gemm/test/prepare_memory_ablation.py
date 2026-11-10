from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
s=(root/'src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp').read_text()
s=s.replace('#include "gemm.h"','#include "../src/gemm.h"')
variants=[s]
no_load=re.sub(r'^\s*(?:a_reg[01]|bv[0-7]) = vld1q_f32\([^;]+;','',s,flags=re.M)
regs=['a_reg0','a_reg1']+[f'bv{i}' for i in range(8)]
init='\n    // Diagnostic only: fixed inputs, made opaque to prevent constant folding.\n'
for i,r in enumerate(regs): init+=f'    {r} = vdupq_n_f32({i+1}.0f / 128.0f);\n'
init+='    // Empty compiler barrier emits no assembly instructions.\n'
init+='    asm volatile("" : '+', '.join(f'"+w"({r})' for r in regs)+');\n'
no_load=no_load.replace('    int kr = 0;', '    int kr = 0;'+init)
variants.append(no_load)
start=s.index('    cp = C;\n    vst1q_f32')
acc=[f'cv{i}{j}' for i in range(4) for j in range(4)]
no_store=s[:start]+'    // Keep all arithmetic live without emitting a result store.\n'
no_store+='    // Empty compiler barrier: no runtime instructions.\n'
no_store+='    asm volatile("" : : '+', '.join(f'"w"({r})' for r in acc)+' : "memory");\n}\n'
variants.append(no_store)
for i,v in enumerate(variants): (root/f'test/.memory_kernel{i}.cpp').write_text(v)
