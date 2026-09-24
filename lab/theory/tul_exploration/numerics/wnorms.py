import torch, sys, re, collections, json
out={}
for name in sys.argv[1:]:
    sd=torch.load(f"/home/wolfe/morph-to/checkpoints/morph/{name}",map_location="cpu",mmap=True,weights_only=False)
    m=sd["model"]
    if name==sys.argv[1]:
        ks=[k for k in m if k.startswith("core.0.")]
        print("core.0 keys:",[ (k,tuple(m[k].shape)) for k in ks])
    agg=collections.OrderedDict()
    for k,t in m.items():
        mm=re.match(r"(prelude|core|coda)\.(\d+)\.(.*)",k)
        if not mm or t.dim()<2: continue
        sec,i,rest=mm.groups()
        # group by section + param role
        role=re.sub(r"\d+","#",rest)
        key=(sec,int(i),role)
        agg[key]=t.float().norm().item()/ (t.numel()**0.5)
    out[name]=agg
    # print summary: rms per section per role
    per=collections.defaultdict(list)
    for (sec,i,role),v in agg.items(): per[(sec,role)].append(v)
    print("==",name)
    for (sec,role),vs in sorted(per.items()):
        print(f"  {sec:7s} {role:60s} rms-per-elem mean {sum(vs)/len(vs):.5f}  n={len(vs)}")
