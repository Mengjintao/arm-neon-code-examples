from pathlib import Path
r = Path(__file__).resolve().parents[2]
o = r / '.ablation'
o.mkdir(exist_ok=True)
k = (r/'src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp').read_text()
driver = (r/'src/main.cpp').read_text().split('\nint main(')[0]
step = (r/'step/register_lane.cpp').read_text()
body = step.split('for (int k = 0; k < K; k++) {')[1].split('\n            }')[0]
body = body.replace('bp += ldb;', 'bp += 16;')
head = k[:k.index('    float32x4_t bv0,')]
store = k[k.index('    cp = C;\n    vst1q_f32'):]

def plain(unroll, prefetch):
    hint = '''
        if (next_A && (kr & 3) == 0) __builtin_prefetch(next_A + kr * 4, 0, 3);
        if (next_B) __builtin_prefetch(next_B + kr * 16, 0, 3);
''' if prefetch else ''
    one = body + '\n        ++kr;\n'
    group_hint = ('''
        if (next_B) {
            __builtin_prefetch(next_B + kr * 16, 0, 3);
            __builtin_prefetch(next_B + (kr + 1) * 16, 0, 3);
        }
        if (next_A && (kr & 3) == 0)
            __builtin_prefetch(next_A + kr * 4, 0, 3);
''' if prefetch else '')
    return head + '    float32x4_t av, bv0, bv1, bv2, bv3;\n    int kr = 0;\n' + (f'    for (; kr + {unroll-1} < k_len;) {{\n' + group_hint + one*unroll + '    }\n' if unroll>1 else '') + '    for (; kr < k_len;) {\n' + hint + one + '    }\n' + store

# name, kernel source, A prefetch, B prefetch, C prefetch, pack implementation
cases = [
 ('V0_register', plain(1,False),0,0,0,'scalar'),
 ('V1_unroll2', plain(2,False),0,0,0,'scalar'),
 ('V2_pipeline',k,0,0,0,'scalar'),
 ('V3_prefetch',k,1,1,1,'scalar'),
 ('V4_vector_pack',k,1,1,1,'vector'),
 ('drop_A_prefetch',k,0,1,1,'vector'),
 ('drop_B_prefetch',k,1,0,1,'vector'),
 ('drop_C_prefetch',k,1,1,0,'vector'),
 ('drop_all_prefetch',k,0,0,0,'vector'),
 ('drop_pipeline',plain(2,True),1,1,1,'vector'),
 ('plain_unroll4',plain(4,False),0,0,0,'scalar'),
]
for name, src, a, b, c, pack in cases:
    if not a: src=src.replace('if (next_A', 'if (false && next_A')
    if not b: src=src.replace('if (next_B)', 'if (false && next_B)')
    d=driver if c else driver.replace('if (next_jr < nc)', 'if (false && next_jr < nc)')
    (o/f'{name}_kernel.cpp').write_text(src.replace('"gemm.h"','"../src/gemm.h"'))
    (o/f'{name}_driver.cpp').write_text(d.replace('"gemm.h"','"../src/gemm.h"')+'\nextern "C" void run(int m,int n,int k,const float*a,int lda,const float*b,int ldb,float*c,int ldc,float*pa,float*pb) { gemm(m,n,k,a,lda,b,ldb,c,ldc,pa,pb); }\n')
(o/'cases.txt').write_text(''.join(f'{name} {pack}\n' for name,_,_,_,_,pack in cases))
