import re,subprocess
from pathlib import Path
valid=True
for v in (6,7):
 s=subprocess.check_output(['objdump','-d',f'.time_components/kernel{v}.o'],text=True)
 ins=[]
 for line in s.splitlines():
  m=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]{8}\s+(\S+)\s*(.*)',line)
  if m:ins.append((int(m[1],16),m[2],m[3]))
 loops=[]
 for a,o,args in ins:
  if o.startswith('b.'):
   try:dest=int(args.split()[0],16)
   except ValueError:continue
   if dest<a:
    body=[x for x in ins if dest<=x[0]<=a]
    if sum(x[1]=='fmla' for x in body)==32:loops.append(body)
 print('CODEGEN',v,'32-FMLA-loops',len(loops))
 if v==6:
  assert loops,'No live computation'
  assert all('[sp' in x[2] for x in ins if x[1].startswith('st')),'Unexpected output store'
 else:
  valid=bool(loops) and all(not any(x[1].startswith(('ld','st')) for x in b) for b in loops)
  print('NO_AB_LOAD usable=',valid,'loop memory ops=',[[x for x in b if x[1].startswith(('ld','st'))] for b in loops])
Path('.time_components/variant_count').write_text('8' if valid else '7')
