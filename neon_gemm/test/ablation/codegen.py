from pathlib import Path
import subprocess,re
for line in Path('.ablation/cases.txt').read_text().splitlines():
    name=line.split()[0]
    s=subprocess.check_output(['objdump','-d','--disassemble=neon_4x16_next_AB(int, float const*, float const*, float*, int, float const*, float const*)','-C',f'.ablation/{name}.so'],text=True)
    ins=[]
    for l in s.splitlines():
        m=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]+\s+(.*)',l)
        if m: ins.append((int(m[1],16),m[2]))
    loops=[]
    for addr,t in ins:
        m=re.search(r'\b(?:b\.[a-z]+|cbnz|cbz|tbnz|tbz)\s+.*?\b([0-9a-f]+)\s+<',t)
        if not m: continue
        target=int(m[1],16)
        if target>=addr: continue
        body=[x for a,x in ins if target<=a<=addr]
        fma=sum('fmla' in x for x in body)
        if not fma: continue
        spills=[x for x in body if '[sp' in x and re.search(r'\b[qd][0-9]+\b',x)]
        loops.append((fma,len(body),len(spills),sum('prfm' in x for x in body)))
    print('CODEGEN',name,'K loops (FMLA,instructions,vector stack accesses,PRFM)=',loops)
    if not loops: raise RuntimeError('No arithmetic loop found: '+name)
