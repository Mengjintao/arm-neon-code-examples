# 消融实验（第 30 页，一张表）：按幻灯片的讲法顺序累加，每一行只比上一行多改一处。
#   1–5  教学代码：step/naive → ikj → cache → cache_inline_pack → 本目录 cache_pack_kmn（Kc→Mc→Nc）
#   6–12 src/main.cpp 的驱动（pc→ic→jc→jr→ir）：微内核 → K 展开 → 流水 → 双缓冲 → NEON 打包 → 分块参数 → 预取
# 另有旁支 X_*（第 8、23 页要用，不进第 30 页的表）。
# 生成 .ablation3/ 下每个版本的源文件和 cases.txt（名字 重复次数 源文件… | 额外编译选项），src/、step/ 不改。
from pathlib import Path

r = Path(__file__).resolve().parents[2]
o = r / '.ablation3'
o.mkdir(exist_ok=True)
RUN = ('extern "C" void run(int m,int n,int k,const float*a,int lda,const float*b,int ldb,float*c,int ldc,'
       'float*pa,float*pb)')

def step(name, path, fn):
    src = (f'#include <cstring>\n#define main step_original_main\n#include "{path}"\n#undef main\n'
           f'{RUN} {{ {fn}(m,n,k,(float*)a,lda,(float*)b,ldb,c,ldc); }}\n')
    (o / f'{name}.cpp').write_text(src)
    return [f'.ablation3/{name}.cpp']

# 分块 128³ 的 gemm.h 副本（只改 MC/NC/KC 三个常量）
h = (r / 'src/gemm.h').read_text()
h128 = h.replace('constexpr int MC = 88;', 'constexpr int MC = 128;').replace(
    'constexpr int NC = 48;', 'constexpr int NC = 128;').replace('constexpr int KC = 256;', 'constexpr int KC = 128;')
assert h128.count('= 128;') == 3
(o / 'gemm_128.h').write_text(h128)

main = (r / 'src/main.cpp').read_text().split('\nint main(')[0]
shim = ('\nvoid neon_4x16_next_AB(int K, const float* A, const float* B, float* C, int ldc,'
        ' const float*, const float*) { neon_4x16(K, A, B, C, ldc); }\n')

def gemm(name, kernel, pack, blocks128, prefetch):
    d = main.replace('"gemm.h"', '"gemm_128.h"' if blocks128 else '"../src/gemm.h"')
    if not prefetch:
        d = d.replace('if (next_jr < nc)', 'if (false && next_jr < nc)')
        assert 'false &&' in d
    (o / f'{name}_driver.cpp').write_text(d + f'\n{RUN} {{ gemm(m,n,k,a,lda,b,ldb,c,ldc,pa,pb); }}\n')
    k = (r / 'src' / kernel).read_text().replace('"gemm.h"', '"../src/gemm.h"')
    if 'neon_4x16_next_AB(' not in k:
        k += shim
    (o / f'{name}_kernel.cpp').write_text(k)
    files = [f'.ablation3/{name}_driver.cpp', f'.ablation3/{name}_kernel.cpp']
    if pack == 'scalar':
        return files + ['src/pack_s.cpp', '|', '-fno-tree-vectorize']
    if pack == 'neon':
        return files + ['src/pack_v.cpp']
    return files + ['src/pack_v_buffer.cpp']

S = r / 'step'
K1, K2, K3, K4, K5 = ('neon_4x16.cpp', 'neon_4x16_2unroll.cpp', 'neon_4x16_2unroll_pipeline.cpp',
                      'neon_4x16_2unroll_pipeline_2buffer.cpp', 'neon_4x16_2unroll_pipeline_2buffer_prfm.cpp')
# 名字, 每轮计时次数, 源文件
cases = [
    ('L01_ijk', 1, step('L01_ijk', S / 'naive.cpp', 'naive')),
    ('L02_ikj', 5, step('L02_ikj', S / 'ikj.cpp', 'ikj')),
    ('L03_cache', 5, step('L03_cache', S / 'cache.cpp', 'cache_block')),
    ('L04_pack', 5, step('L04_pack', S / 'cache_inline_pack.cpp', 'cache_block')),
    ('L05_kmn', 5, step('L05_kmn', r / 'test/ablation3/cache_pack_kmn.cpp', 'cache_block')),
    ('L06_register', 5, gemm('L06_register', K1, 'scalar', True, False)),
    ('L07_unroll2', 5, gemm('L07_unroll2', K2, 'scalar', True, False)),
    ('L08_pipeline', 5, gemm('L08_pipeline', K3, 'scalar', True, False)),
    ('L09_2buffer', 5, gemm('L09_2buffer', K4, 'scalar', True, False)),
    ('L10_neon_pack', 5, gemm('L10_neon_pack', K4, 'vector', True, False)),
    ('L11_blocks', 5, gemm('L11_blocks', K4, 'vector', False, False)),
    ('L12_prefetch', 5, gemm('L12_prefetch', K5, 'vector', False, True)),
    # 旁支（不进阶梯）：第 23 页的中间一步，第 9 行只把打包换成 src/pack_v.cpp（NEON，无双缓冲）
    ('X_neon_pack', 5, gemm('X_neon_pack', K4, 'neon', True, False)),
]
(o / 'cases.txt').write_text(''.join(f'{n} {reps} {" ".join(f)}\n' for n, reps, f in cases))
print('prepared', [c[0] for c in cases])
