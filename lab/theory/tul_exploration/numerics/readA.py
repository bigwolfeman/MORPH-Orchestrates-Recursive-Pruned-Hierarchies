import torch, sys, math
for name in sys.argv[1:]:
    p=f"/home/wolfe/morph-to/checkpoints/morph/{name}"
    sd=torch.load(p,map_location="cpu",mmap=True,weights_only=False)
    m=sd.get("model",sd.get("model_state_dict",sd))
    if not isinstance(m,dict): print(type(m)); continue
    ks=[k for k in m if "injection" in k or "input_norm" in k]
    print("==",name, "top keys", list(sd.keys())[:10] if isinstance(sd,dict) else None)
    for k in ks:
        t=m[k].float()
        print(" ",k,tuple(t.shape), "mean",t.mean().item(),"min",t.min().item(),"max",t.max().item())
    if any(k.endswith("injection.log_A") for k in m):
        k=[k for k in m if k.endswith("injection.log_A")][0]
        A=m[k].float().exp().clamp(max=0.9999)
        C=1024; nctx=A.numel()
        g=math.sqrt((C-nctx + (A**2).sum().item())/C)
        print("  ctx dims",nctx,"mean A",A.mean().item(),"rms A",A.pow(2).mean().sqrt().item(),"decay-only floor gain",g)
        kdt=[k for k in m if k.endswith("injection.log_dt")][0]
        dt=m[kdt].float().exp(); print("  dt mean",dt.mean().item(),"min",dt.min().item(),"max",dt.max().item())
