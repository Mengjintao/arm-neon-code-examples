#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec > result.out 2> error.err
trap 'rm -f test/.memory_*.o test/.memory_kernel[0-2].cpp test/.memory_ablation' EXIT
python3 test/prepare_memory_ablation.py
for v in 0 1 2; do
 g++ -O3 -fno-schedule-insns -std=c++17 -Dgemm=gemm$v -Dmain=unused_main$v -Dneon_4x16_next_AB=kernel$v -c src/main.cpp -o test/.memory_main$v.o
 g++ -O3 -fno-schedule-insns -fno-tree-pre -fno-tree-fre -fno-gcse -fno-move-loop-invariants -fno-tree-loop-im -std=c++17 -Dneon_4x16_next_AB=kernel$v -c test/.memory_kernel$v.cpp -o test/.memory_kernel$v.o
done
g++ -O3 -fno-schedule-insns -std=c++17 test/memory_ablation.cpp src/pack_v_buffer.cpp test/.memory_*.o -o test/.memory_ablation
python3 - <<'PY'
import re,subprocess
for v in range(3):
 s=subprocess.check_output(['objdump','-d',f'test/.memory_kernel{v}.o'],text=True)
 ins=[]
 for line in s.splitlines():
  m=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]{8}\s+(\S+)\s*(.*)',line)
  if m:ins.append((int(m[1],16),m[2],m[3]))
 fmla=[i for i in ins if i[1]=='fmla']
 assert len(fmla)>=64, 'Arithmetic removed'
 loops=[]
 for addr,op,args in ins:
  if op.startswith('b.'):
   target=int(args.split()[0],16)
   if target<addr:
    body=[i for i in ins if target<=i[0]<=addr]
    if sum(i[1]=='fmla' for i in body)>=32:loops.append(body)
 assert loops, 'No FMLA loop'
 body=min(loops,key=len)
 loads=[i for i in body if i[1].startswith('ld')]
 stores=[i for i in ins if i[1].startswith('st')]
 print('CODEGEN',v,'static_FMLA=',len(fmla),'loop_FMLA=',sum(i[1]=='fmla' for i in body),'loop_loads=',len(loads),'loop_PRFM=',sum(i[1]=='prfm' for i in body))
 assert sum(i[1]=='fmla' for i in body)==32, 'Changed hot-loop FMLA count'
 if v==1:
  steady=[b for b in loops if sum(i[1]=='fmla' for i in b)==32]
  assert all(not any(i[1].startswith(('ld','st')) for i in b) for b in steady), 'Memory access in a no-load loop'
  assert not any('[sp' in args and re.match(r'q[0-9]',args) for _,op,args in ins if op.startswith(('ld','st'))), 'Vector spill outside loop'
  print('All 32-FMLA loop paths checked:',len(steady))
 if v==2:
  assert all('[sp' in i[2] for i in stores), f'Unexpected non-stack stores: {stores}'
 print('STORE instructions:',[(op,args) for _,op,args in stores])
 print('CODEGEN PASS')
PY
taskset -c 0 test/.memory_ablation
