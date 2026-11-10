from pathlib import Path
import re
root=Path(__file__).resolve().parents[2]
out=root/'.time_components'
out.mkdir(exist_ok=True)
s=(root/'src/main.cpp').read_text();s=s[:s.index('\nint main(')]
s=s.replace('#include "gemm.h"','#include "../src/gemm.h"')
# All cached panels have distinct storage: four K blocks, twelve M blocks.
for v in range(8):
 t=s.replace('void gemm(',f'void gemm{v}(')
 if v in (1,2,3,4):
  t=t.replace('        pack_B(kc, N, B + (size_t)pc * ldb, ldb, packed_B);',
   '        float* pb = packed_B + (size_t)(pc / KC) * KC * ((N + NR - 1) / NR * NR);\n'+
   ('' if v in (3,4) else '        pack_B(kc, N, B + (size_t)pc * ldb, ldb, pb);'))
  t=t.replace('            pack_A(mc, kc, A + (size_t)ic * lda + pc, lda, packed_A);',
   '            float* pa = packed_A + ((size_t)(pc / KC) * ((M + MC - 1) / MC) + ic / MC) * MC * KC;\n'+
   ('' if v in (2,4) else '            pack_A(mc, kc, A + (size_t)ic * lda + pc, lda, pa);'))
  t=t.replace('bp = packed_B +','bp = pb +').replace('ap = packed_A +','ap = pa +').replace('? packed_A : nullptr','? pa : nullptr')
 if v==5:
  t=t.replace('''                            if (pc == 0) {
                                for (int r = 0; r < MR; ++r)
                                    std::fill_n(cp + (size_t)r * ldc, NR, 0.0f);
                            }
''','')
 if v in (6,7):
  name=f'kernel{v}';t=t.replace('neon_4x16_next_AB',name)
  decl=f'void {name}(int,const float*,const float*,float*,int,const float*,const float*);\n'
  t=t.replace('static double gettime()',decl+'\nstatic double gettime()')
 (out/f'driver{v}.cpp').write_text(t)
k=(root/'src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp').read_text().replace('#include "gemm.h"','#include "../src/gemm.h"')
start=k.index('    cp = C;\n    vst1q_f32')
acc=[f'cv{i}{j}' for i in range(4) for j in range(4)]
t=k[:start]+'    // Empty operand sink preserves arithmetic; emits no machine instruction.\n    asm volatile("" : : '+', '.join(f'"w"({r})' for r in acc)+' : "memory");\n}\n'
(out/'kernel6.cpp').write_text(t.replace('neon_4x16_next_AB','kernel6'))
t=re.sub(r'^\s*(?:a_reg[01]|bv[0-7]) = vld1q_f32\([^;]+;','',k,flags=re.M)
regs=['a_reg0','a_reg1']+[f'bv{i}' for i in range(8)]
init='\n'+''.join(f'    {r} = vdupq_n_f32({i+1}.0f / 128.0f);\n' for i,r in enumerate(regs))
init+='    asm volatile("" : '+', '.join(f'"+w"({r})' for r in regs)+');\n'
t=t.replace('    int kr = 0;', '    int kr = 0;'+init)
(out/'kernel7.cpp').write_text(t.replace('neon_4x16_next_AB','kernel7'))
