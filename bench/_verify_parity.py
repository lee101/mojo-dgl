"""Parity + input-identity check for the 16 bench cases, at the bench shape (1M/32d).

For each case: run mojodgl and upstream dgl on the EXACT same numpy arrays
(mojo gets numpy, dgl gets torch.as_tensor of the same buffer), compare outputs,
and report max abs diff. Also checks the bench lambdas pass identical inputs
and that no case skips work on one side (output shape + value check).
"""
import os, sys, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import numpy as np
import mojodgl as mdgl
import mojodgl.function as fn
import mojodgl.ops as ops
import dgl as udgl
import dgl.function as ufn
import dgl.ops as uops
import torch

NODES, EDGES, DIM = 200_000, 1_000_000, 32

def t(a): return torch.as_tensor(a)

def check(name, ours, theirs, tol=1e-9):
    o = np.asarray(ours)
    th = np.asarray(theirs.detach().cpu().numpy() if hasattr(theirs, "detach") else theirs)
    assert o.shape == th.shape, f"{name}: shape {o.shape} vs {th.shape}"
    md = float(np.max(np.abs(o - th))) if o.size else 0.0
    ok = np.allclose(o, th, rtol=1e-9, atol=tol)
    print(f"{name:55} shape={str(o.shape):18} maxabsdiff={md:.3e} allclose(1e-9)={ok}")
    return ok

rng = np.random.default_rng(0)
u = np.ascontiguousarray(rng.integers(0, NODES, size=EDGES))
v = np.ascontiguousarray(rng.integers(0, NODES, size=EDGES))
x = np.ascontiguousarray(rng.normal(size=(NODES, DIM)))
e = np.ascontiguousarray(rng.normal(size=(EDGES, DIM)))
z = np.ascontiguousarray(rng.normal(size=(EDGES, 1)))
zh = np.ascontiguousarray(rng.normal(size=(EDGES, 8, 1)))
g = mdgl.graph((u, v)); g.ndata["h"] = x; g.edata["e"] = e
ug = udgl.graph((torch.as_tensor(u), torch.as_tensor(v))); ug.ndata["h"] = torch.as_tensor(x); ug.edata["e"] = torch.as_tensor(e)
print(f"graph built: mojodgl nodes={g.number_of_nodes()} edges={g.number_of_edges()}  upstream nodes={ug.number_of_nodes()} edges={ug.number_of_edges()}")
print(f"same edge arrays: u/v shared numpy vs torch.from same u/v -> identical by construction; x/e/z/zh same buffers")

all_ok = True
for nm, f1, f2 in [
 ("u_mul_e_sum", lambda: ops.u_mul_e_sum(g,x,e), lambda: uops.u_mul_e_sum(ug,t(x),t(e))),
 ("u_add_e_mean", lambda: ops.u_add_e_mean(g,x,e), lambda: uops.u_add_e_mean(ug,t(x),t(e))),
 ("u_div_e_min(+4 shift like bench)", lambda: ops.u_div_e_min(g,x+4.0,e+4.0), lambda: uops.u_div_e_min(ug,t(x+4.0),t(e+4.0))),
 ("copy_u_max", lambda: ops.copy_u_max(g,x), lambda: uops.copy_u_max(ug,t(x))),
 ("copy_e_mean", lambda: ops.copy_e_mean(g,e), lambda: uops.copy_e_mean(ug,t(e))),
 ("u_dot_e", lambda: ops.u_dot_e(g,x,e), lambda: uops.u_dot_e(ug,t(x),t(e))),
 ("u_mul_v", lambda: ops.u_mul_v(g,x,x), lambda: uops.u_mul_v(ug,t(x),t(x))),
 ("edge_softmax dst", lambda: ops.edge_softmax(g,z), lambda: uops.edge_softmax(ug,t(z))),
 ("edge_softmax src", lambda: ops.edge_softmax(g,z,norm_by="src"), lambda: uops.edge_softmax(ug,t(z),norm_by="src")),
 ("edge_softmax 8 heads", lambda: ops.edge_softmax(g,zh), lambda: uops.edge_softmax(ug,t(zh))),
]:
    all_ok &= check(nm, f1(), f2())

# segment
rng2 = np.random.default_rng(11)
cuts = np.sort(rng2.choice(EDGES, size=199_999, replace=False))
seglen = np.diff(np.concatenate([[0], cuts, [EDGES]])).astype(np.int64)
sv = np.ascontiguousarray(rng2.normal(size=(EDGES, DIM)))
for nm, red in [("segment_reduce sum","sum"),("segment_reduce max","max")]:
    all_ok &= check(nm, ops.segment_reduce(seglen,sv,red), uops.segment_reduce(torch.as_tensor(seglen), t(sv), red))
all_ok &= check("segment_softmax", ops.segment_softmax(seglen,sv), uops.segment_softmax(torch.as_tensor(seglen), t(sv)), tol=1e-9)

# gather_mm
rng3 = np.random.default_rng(12)
a = np.ascontiguousarray(rng3.normal(size=(EDGES, 16)))
b = np.ascontiguousarray(rng3.normal(size=(64, 16, 16)))
idx = np.ascontiguousarray(rng3.integers(0, 64, size=EDGES))
all_ok &= check("gather_mm", ops.gather_mm(a,b,idx_b=idx), uops.gather_mm(t(a),t(b),idx_b=torch.as_tensor(idx)), tol=1e-9)

# update_all
g.update_all(fn.u_mul_e("h","e","m"), fn.sum("m","z"))
ug.update_all(ufn.u_mul_e("h","e","m"), ufn.sum("m","z"))
all_ok &= check("update_all u_mul_e+sum", g.ndata["z"], ug.ndata["z"])
g.update_all(fn.copy_u("h","m"), fn.max("m","z"))
ug.update_all(ufn.copy_u("h","m"), ufn.max("m","z"))
all_ok &= check("update_all copy_u+max", g.ndata["z"], ug.ndata["z"])

print("\nALL PARITY OK" if all_ok else "\nPARITY FAILURES FOUND")
# Work-skip audit notes (static, verified by reading bench.py + ops/*):
print("""
Work-skip audit (static reading of bench/bench.py and python/mojodgl/ops/*.py):
- All 16 cases: mojodgl gets (graph, numpy arrays), upstream gets (graph, torch.as_tensor of SAME numpy buffer).
  No case pre-computes, caches, or short-circuits one side; both lambdas call the public op each time.
- data()/segment_data()/mm_data() are cached (_CACHE) but identical across calls - caching is input sharing, not work skipping.
- t(a)=torch.as_tensor(a) inside the timed upstream lambda shares memory (no copy) - same as mojodgl reading the numpy buffer.
- update_all cases call the full update_all on both sides each rep; output read after.
- warmup: only mojodgl is warmed (ours() once); upstream is NOT warmed before timing - if anything this favours mojodgl slightly.
  First upstream rep may include JIT/lazy init. Effect is small (best-of-3) and documented here, not corrected (bench.py untouched).
- timeit = best of 3, sequential (mojo first, then dgl), under flock /tmp/mojo-bench.lock.
""")
