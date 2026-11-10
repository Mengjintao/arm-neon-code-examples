#!/usr/bin/env bash
# 时间组成（第 29 页）：E2E 拆成 打包 B / 打包 A / 微内核 / 其他，微内核再拆 FMLA / load / store。
# 结果写 test/time.out；src/ 不改，各版本微内核由 prepare_kernels.py 生成到临时目录。
set -euo pipefail
cd "$(dirname "$0")/.."
exec > test/time.out 2> test/time.err
BUILD=test/.build_time
rm -rf "$BUILD"; mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT

FLAGS="-O3 -mcpu=tsv110 -std=c++17"
PACK=${PACK:-src/pack_v_buffer.cpp}
echo "g++ $(g++ -dumpfullversion); $FLAGS; CPU0; 1024^3; main.cpp + $PACK + neon_4x16_2unroll_pipeline_2buffer_prfm.cpp"
python3 test/prepare_kernels.py "$BUILD"
# 去掉 load 后 GCC 会把 lane 广播外提成 dup，寄存器不够而溢出到栈；
# 所以 ref/noload/nostore/pure 统一加防外提选项，full 和 empty 用正式选项。
ABL="-fno-tree-pre -fno-tree-fre -fno-gcse -fno-move-loop-invariants -fno-tree-loop-im"
echo "ablation kernels (ref noload nostore pure) add: $ABL"
for v in full ref noload nostore pure empty; do
  g++ $FLAGS -Dgemm=gemm_$v -Dmain=unused_main_$v -Dneon_4x16_next_AB=kern_$v \
      -Dpack_A=timed_pack_A -Dpack_B=timed_pack_B -c src/main.cpp -o "$BUILD/main_$v.o"
  case $v in full|empty) KF="";; *) KF="$ABL";; esac
  g++ $FLAGS $KF -Dneon_4x16_next_AB=kern_$v -c "$BUILD/kern_$v.cpp" -o "$BUILD/kern_$v.o"
done
g++ $FLAGS -c "$PACK" -o "$BUILD/pack.o"
g++ $FLAGS -c test/time_split.cpp -o "$BUILD/time_split.o"
g++ "$BUILD"/*.o -o "$BUILD/time_split"

# 机器码检查：K 循环里的 FMLA 条数、load/store、栈溢出
python3 - "$BUILD" <<'PY'
import re, subprocess, sys
b = sys.argv[1]
for v in ['full', 'ref', 'noload', 'nostore', 'pure']:
    s = subprocess.check_output(['objdump', '-d', f'{b}/kern_{v}.o'], text=True)
    ins = []
    for line in s.splitlines():
        m = re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]{8}\s+(\S+)\s*(.*)', line)
        if m: ins.append((int(m[1], 16), m[2], m[3]))
    loops = []
    for addr, op, args in ins:
        if op.startswith('b.') or op in ('cbnz', 'cbz'):
            t = re.search(r'\b([0-9a-f]+) <', args)
            if t and int(t[1], 16) < addr:
                body = [i for i in ins if int(t[1], 16) <= i[0] <= addr]
                if sum(i[1] == 'fmla' for i in body) >= 32: loops.append(body)
    body = min(loops, key=len)
    ld = [i for i in body if i[1].startswith('ld')]
    st = [i for i in ins if i[1].startswith('st') and '[sp' not in i[2]]
    spill = [i for i in ins if i[1].startswith(('ld', 'st')) and '[sp' in i[2] and re.search(r'\bq\d+', i[2])]
    print(f'CODEGEN {v:8s} total_FMLA={sum(i[1]=="fmla" for i in ins)} loop_FMLA={sum(i[1]=="fmla" for i in body)} '
          f'loop_instr={len(body)} loop_loads={len(ld)} loop_PRFM={sum(i[1]=="prfm" for i in body)} '
          f'C_stores={len(st)} q_stack={len(spill)}')
PY
taskset -c 0 "$BUILD/time_split"
