import sys, re, json, os, statistics
SP=os.path.dirname(os.path.abspath(__file__))
tag=sys.argv[1]
lines=open(f"{SP}/logs/{tag}.log", errors="replace").read().splitlines()
PAT=re.compile(r"\[dbg\] step (\d+): ([0-9.]+)s")
ts={}
for l in lines:
    m=PAT.search(l)
    if m: ts[int(m.group(1))]=float(m.group(2))
pk=[float(m.group(1)) for m in (re.search(r"peak=([0-9.]+)GB", l) for l in lines) if m]
rsv=[float(m.group(1)) for m in (re.search(r"reserved[= ]([0-9.]+)", l) for l in lines) if m]
steady=[v for k,v in sorted(ts.items()) if k>=20]
out={"tag":tag,"n_steady":len(steady)}
if steady:
    out["mean_s"]=round(statistics.mean(steady),4)
    out["median_s"]=round(statistics.median(steady),4)
    out["sps"]=round(1/statistics.mean(steady),4)
    out["sd_s"]=round(statistics.pstdev(steady),4)
if pk: out["peak_GB"]=max(pk)
j=f"{SP}/probe/{tag}.jsonl"
if os.path.exists(j):
    rows=[json.loads(x) for x in open(j) if x.strip()]
    g=lambda n: next((r["loss/total"] for r in rows if r["step"]==n), None)
    out["loss0"]=g(0); out["loss1"]=g(1); out["loss20"]=g(20); out["loss59"]=g(59)
os.makedirs(f"{SP}/summ",exist_ok=True)
json.dump(out,open(f"{SP}/summ/{tag}.json","w"),indent=1)
print(json.dumps(out))
json.dump({str(k):v for k,v in ts.items()}, open(f"{SP}/summ/{tag}_steps.json","w"))
