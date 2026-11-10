# 由 src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp 生成"去掉一类指令"的微内核（只用于拆时间，结果不是正确的 GEMM）。
#   noload : 去掉 K 循环里 A、B 的 vld1q，输入寄存器开头赋一次固定值
#   nostore: 去掉最后 16 条 C 的 vst1q，用空的编译器屏障保住累加结果
#   ref    : 和 full 同一份源码，但与 noload/nostore/pure 用同一组防外提编译选项（作对照）
#   pure   : 两者都去掉，只剩 FMLA（和开头的 C load、预取）
#   empty  : 空函数，测内核以外的部分（打包、C 清零、循环、C 预取、边界）
import re, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
s = (root / 'src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp').read_text().replace('#include "gemm.h"', f'#include "{root}/src/gemm.h"')

def no_load(src):
    src = re.sub(r'^\s*(?:a_reg[01]|bv[0-7]) = vld1q_f32\([^;]+;', '', src, flags=re.M)
    regs = ['a_reg0', 'a_reg1'] + [f'bv{i}' for i in range(8)]
    init = '\n'.join(f'    {r} = vdupq_n_f32({i + 1}.0f / 128.0f);' for i, r in enumerate(regs))
    init += '\n    asm volatile("" : ' + ', '.join(f'"+w"({r})' for r in regs) + ');\n'
    return src.replace('    int kr = 0;', '    int kr = 0;\n' + init)

def no_store(src):
    start = src.index('    cp = C;\n    vst1q_f32')
    acc = [f'cv{i}{j}' for i in range(4) for j in range(4)]
    return src[:start] + '    asm volatile("" : : ' + ', '.join(f'"w"({r})' for r in acc) + ' : "memory");\n}\n'

variants = {'full': s, 'ref': s, 'noload': no_load(s), 'nostore': no_store(s), 'pure': no_store(no_load(s))}
for name, src in variants.items():
    (out / f'kern_{name}.cpp').write_text(src)
(out / 'kern_empty.cpp').write_text(
    'void neon_4x16_next_AB(int, const float*, const float*, float*, int, const float*, const float*) {}\n')
print('generated', ', '.join(list(variants) + ['empty']))
