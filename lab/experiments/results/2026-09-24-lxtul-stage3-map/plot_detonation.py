"""Arm A (nofp) detonation and recovery vs arm C, from probe.jsonl and the run logs."""
import json, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S="/home/wolfe/morph-scratch/lxtul-s3"
def probe(arm):
    d={}
    for l in open(f"{S}/{arm}/probe.jsonl"):
        try: r=json.loads(l)
        except Exception: continue
        d.setdefault("step",[]).append(r["step"])
        for k in ("preclip/total","loss/total","loss/gain_reg_weighted","loss/gain_slot_max"):
            d.setdefault(k,[]).append(r.get(k,np.nan))
    d={k:np.array(v,float) for k,v in d.items()}
    d["obj"]=d["loss/total"]-np.nan_to_num(d["loss/gain_reg_weighted"])
    return d
def val(arm):
    t=open(f"{S}/run_{arm}.log").read()
    m=re.findall(r"\[VAL\s+(\d+)\] loss=([\d.]+)",t)
    return np.array([int(a) for a,_ in m]),np.array([float(b) for _,b in m]),float(re.search(r"Final val_loss=([\d.]+)",t).group(1))
A,C=probe("nofp"),probe("map"); va,vc=val("nofp"),val("map")
i=int(np.nanargmax(A["preclip/total"])); spike=int(A["step"][i])
BLUE,ORANGE="#0072B2","#E69F00"
def med(x,w=51): return np.array([np.nanmedian(x[max(0,k-w//2):k+w//2+1]) for k in range(len(x))])
fig,ax=plt.subplots(4,2,figsize=(14,13),gridspec_kw={"width_ratios":[2.2,1]})
Z=(3950,4250)
rows=[("preclip/total","grad norm before clip",True,True),
      ("obj","train objective, no hinge",False,False),
      ("loss/gain_slot_max","max slot gain (bf16)",True,True)]
for r,(k,lab,logy,logz) in enumerate(rows):
    for c,(lo,hi) in enumerate([(0,5000),Z]):
        a=ax[r,c]; mC=(C["step"]>=lo)&(C["step"]<=hi); mA=(A["step"]>=lo)&(A["step"]<=hi)
        a.plot(C["step"][mC],C[k][mC],color=ORANGE,lw=0.6,alpha=0.8,label="arm C: fixed-point term 1.0")
        a.plot(A["step"][mA],A[k][mA],color=BLUE,lw=0.6,alpha=0.9,label="arm A (nofp): term off")
        if k=="obj":
            if c==0:
                a.plot(A["step"],med(A[k]),color=BLUE,lw=1.8,label="arm A, 51-step median")
                a.plot(C["step"],med(C[k]),color=ORANGE,lw=1.8,label="arm C, 51-step median")
            a.set_ylim(10.5,14)
        if k=="preclip/total":
            for y,t in ((1e4,"excursion 1e4"),(1e5,"detonation 1e5")):
                a.axhline(y,color="0.35",ls="--",lw=0.9); a.text(lo+(hi-lo)*0.01,y*1.5,t,fontsize=8,color="0.3")
        if (logy and c==0) or (logz and c==1): a.set_yscale("log")
        a.axvline(spike,color="k",ls=":",lw=0.9); a.set_xlim(lo,hi); a.grid(alpha=0.25)
        if c==0: a.set_ylabel(lab)
        if r==0: a.set_title("full run" if c==0 else f"zoom: steps {Z[0]}-{Z[1]}")
ax[0,0].legend(loc="upper left",fontsize=8); ax[1,0].legend(loc="upper right",fontsize=8)
ax[0,0].annotate(f"step {spike}: {A['preclip/total'][i]:.2g}\n(the hinge penalty; objective\nwithout it: {A['obj'][i]:.2f})",
                 xy=(spike,A['preclip/total'][i]),xytext=(2100,1.5e6),arrowprops=dict(arrowstyle="->"),fontsize=9)
k35=int(np.nanargmax(np.where((A["step"]>3500)&(A["step"]<3700),A["preclip/total"],np.nan)))
ax[0,0].annotate(f"precursor, step {int(A['step'][k35])}: {A['preclip/total'][k35]:.2g}",xy=(A['step'][k35],A['preclip/total'][k35]),
                 xytext=(1300,2e4),arrowprops=dict(arrowstyle="->"),fontsize=9)
a=ax[3,0]
a.plot(vc[0],vc[1],"s-",color=ORANGE,label=f"arm C: final {vc[2]:.4f}")
a.plot(va[0],va[1],"o-",color=BLUE,label=f"arm A (nofp): final {va[2]:.4f}")
a.axvline(spike,color="k",ls=":",lw=0.9); a.set_ylabel("val loss (coda CE)"); a.set_xlabel("step")
a.grid(alpha=0.25); a.legend(fontsize=9); a.set_xlim(0,5000); a.set_ylim(4.25,6.0)
b=ax[3,1]; b.axis("off")
txt=["val loss        arm A    arm C    A - C",""]+[f"step {int(s):5d}:  {x:.4f}   {y:.4f}  {x-y:+.4f}" for s,x,y in zip(va[0],va[1],vc[1]) if s>=3000]
txt+=["",f"final:       {va[2]:.4f}   {vc[2]:.4f}  {va[2]-vc[2]:+.4f}"]
b.text(0,1,"\n".join(txt),va="top",family="monospace",fontsize=9)
for a in ax[2]: a.set_xlabel("step")
fig.suptitle(f"LXTUL Stage 3 arm A (fixed-point term off): one detonation at step {spike} and its recovery",fontsize=13)
fig.tight_layout(rect=(0,0,1,0.97)); fig.savefig(f"{S}/nofp_detonation.png",dpi=130); print("saved")
