#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec > result.out 2> error.err
trap 'rm -f test/.fmla_lane_ipc' EXIT
uname -a
g++ --version | head -1
g++ -O2 -fno-tree-pre -fno-tree-fre -fno-gcse -fno-move-loop-invariants -fno-tree-loop-im \
 -std=c++17 -march=armv8-a+simd test/fmla_lane_ipc.cpp -o test/.fmla_lane_ipc
python3 - <<'PY'
import subprocess,re
s=subprocess.check_output(['objdump','-d','--disassemble=_Z9fmla_laneiPKfS0_S0_Pf','test/.fmla_lane_ipc'],text=True)
ins=[]
for line in s.splitlines():
 m=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]{8}\s+(\S+)\s*(.*)',line)
 if m: ins.append((int(m[1],16),m[2],m[3]))
loops=[]
for addr,op,args in ins:
 if op=='b.ne':
  target=int(args.split()[0],16)
  if target<addr: loops.append([i for i in ins if target<=i[0]<=addr])
assert len(loops)==1, 'Expected one hot loop'
body=loops[0]; fmla=[i for i in body if i[1]=='fmla']
assert len(fmla)==96, f'FMLA count {len(fmla)}'
assert all(re.search(r'v\d+\.s\[[0-3]\]',i[2]) for i in fmla), 'Not lane FMLA'
assert len({i[2].split(',')[0] for i in fmla})==24
assert not any(i[1].startswith(('ld','st')) for i in body), 'Load/store in hot loop'
print('CODEGEN PASS: 96 lane FMLAs, 24 distinct accumulator registers, no hot-loop load/store')
print('Hot-loop instructions:',len(body))
print('Other hot-loop instructions:',[(op,args) for _,op,args in body if op!='fmla'])
print('Expected retired instructions/sample, hot loop only:',len(body)*200000*20)
PY
for round in 1 2 3; do
 echo "ROUND=$round"
 taskset -c 0 test/.fmla_lane_ipc 0
done
