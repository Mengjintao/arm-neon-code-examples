# 在 test/ablation/prepare.py 生成的 12 个版本之外，再加 4 个"教学代码分块参数 / 循环顺序"下的版本。
# 微内核直接复制 prepare.py 生成的 V0/V1/V2 内核（不改一个字），只换驱动。
from pathlib import Path
import shutil

r = Path(__file__).resolve().parents[2]
o = r / '.ablation'
drv = (r / 'test/ablation2/driver_ipj.cpp').read_text()

# 名字, 复用哪个内核, (Mc, Nc, Kc)
cases = [
    ('S0_register_128', 'V0_register', (128, 128, 128)),
    ('S1_unroll2_128', 'V1_unroll2', (128, 128, 128)),
    ('S2_pipeline_128', 'V2_pipeline', (128, 128, 128)),
    ('S3_pipeline_blocks', 'V2_pipeline', (88, 48, 256)),
]
lines = []
for name, kern, (mc, nc, kc) in cases:
    shutil.copy(o / f'{kern}_kernel.cpp', o / f'{name}_kernel.cpp')
    head = f'#define BMC {mc}\n#define BNC {nc}\n#define BKC {kc}\n'
    (o / f'{name}_driver.cpp').write_text(head + drv)
    lines.append(f'{name} scalar\n')
with open(o / 'cases.txt', 'a') as f:
    f.writelines(lines)
print('added', [c[0] for c in cases])
