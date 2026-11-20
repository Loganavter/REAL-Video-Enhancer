"""Per-shot audit simulation for the reuse gate: does auditing a fraction of reuses keep the violation rate?

Reads (statistic, damage) per reused frame for two provable cuts; a fixed-threshold rule against the served
reference violates on 28.46% of its own calibration clip and 4.88% of the held-out one, and a 10%-audit
controller does not converge at two frames per chunk. Numbers: RU023/RU024 in
reuse-decision-statistic.txt."""
"""Simulate per-shot audit-based certification: does auditing a fraction of reuses keep the violation rate?

Data: per reused frame, (block_maxabs against the reference the gate served, damage share) for both clips.
Policy: process frames in order; every time a chunk of C candidate reuses accumulates, audit a random p of them
(those would have been rendered anyway, so their true damage is known), and move the threshold by a step in the
direction the audited rate demands. Threshold trajectory is continuous across clips: calibrated on A, then the
policy runs on B from A's final threshold without seeing B's labels except through its audits.
"""
import json, random, re
import cv2, numpy as np

def chain_stats(clip, depth_log, damage, block=32):
    depth={}
    for line in open(depth_log, errors="ignore"):
        m=re.search(r"reuse frame=(\d+) depth=(\d+)", line)
        if m: depth[int(m.group(1))]=int(m.group(2))
    d={int(r["frame"]):float(r["share"]) for r in json.load(open(damage))["rows"]}
    cap=cv2.VideoCapture(clip); buf=[]; out=[]; i=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        g=cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        buf.append(g)
        k=depth.get(i)
        if k and len(buf)>k:
            diff=np.abs(g-buf[-1-k]); h,w=diff.shape; hb,wb=h//block,w//block
            v=float(diff[:hb*block,:wb*block].reshape(hb,block,wb,block).mean(axis=(1,3)).max())
            out.append({"frame": i, "x": v, "damage": d.get(i,0.0)})
        i+=1
    cap.release(); return out

A=chain_stats("w1040.mkv","w1040_depth.log","w1040_damage.json")
B=chain_stats("scan2/w960.mkv","scan2/w960_depth.log","scan2/w960_damage.json")
json.dump({"A":A,"B":B}, open("audit_data.json","w"))

def policy(rows, tau, target, p=0.10, chunk=20, seed=7):
    rnd=random.Random(seed); audited=0; violations=0; tau_trace=[]
    i=0
    while i < len(rows):
        block=rows[i:i+chunk]; i+=chunk
        aud=[r for r in block if rnd.random()<p]
        audited+=len(aud)
        if aud:
            rate=sum(1 for r in aud if r["damage"]>0.1)/len(aud)
            tau = tau*1.05 if rate>target else tau*0.98
        for r in block:
            if r["x"]>tau: violations += 1 if r["damage"]>0.1 else 0
        tau_trace.append(round(tau,3))
    n=len(rows)
    return {"n":n,"tau_final":round(tau,3),"violations":violations,
            "violation_rate_pct":round(100*violations/max(n,1),2),"audited":audited,
            "audit_pct":round(100*audited/max(n,1),1),"tau_trace":tau_trace}

for target in (0.005, 0.001):
    ta=policy(A, 2.172, target)                 # start from the Youden point of A
    tb=policy(B, ta["tau_final"], target)       # continue on B from A's final threshold
    print(f"target rate {target*100:.1f}%:")
    print(f"   A: tau {ta['tau_final']}  violations {ta['violations']}/{ta['n']} = {ta['violation_rate_pct']}%  audited {ta['audit_pct']}%")
    print(f"   B: tau {tb['tau_final']}  violations {tb['violations']}/{tb['n']} = {tb['violation_rate_pct']}%  audited {tb['audit_pct']}%")
print("\nno policy, fixed tau=2.172 everywhere:")
for name, rows in (("A",A),("B",B)):
    v=sum(1 for r in rows if r["x"]>2.172 and r["damage"]>0.1)
    print(f"   {name}: {v}/{len(rows)} = {100*v/len(rows):.2f}%")
