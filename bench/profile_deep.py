"""Deeper profiling for bench/analysis.md. Run under the same flock as bench.

pixi run python bench/profile_deep.py   (NOT concurrent with pixi run bench)

Method:
* same arrays/seed as bench/bench.py (NODES=200k, EDGES=1M, DIM=32)
* best-of-5 after 1 warmup, perf_counter; also median for stability
* mojo public entry, upstream dgl, numpy baseline measured on identical arrays
* kernel-only: pre-allocated dst + direct ctypes call (bypasses Python glue)
* glue = public - kernel-only  (includes f64/i64 checks, np.empty, CSR fetch)
* ctypes call count via monkeypatch counter; allocation bytes via tracemalloc-ish
  numpy counting wrapper (np.empty/zeros/repeat/... sizes)
* small-size scaling: 1k/10k/100k edges to expose fixed per-call overhead
"""
import gc, math, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import numpy as np
import mojodgl as dgl
import mojodgl.function as fn
import mojodgl.ops as ops
import dgl as udgl
import dgl.function as ufn
import dgl.ops as uops
import torch
from mojodgl._lib import lib, addr, f64, i64

NODES, EDGES, DIM = 200_000, 1_000_000, 32

def timeit(call, repeat=5):
    call()
    ts=[]
    for _ in range(repeat):
        t0=time.perf_counter(); call(); ts.append(time.perf_counter()-t0)
    return min(ts), float(np.median(ts))

def data():
    rng=np.random.default_rng(0)
    u=np.ascontiguousarray(rng.integers(0,NODES,size=EDGES))
    v=np.ascontiguousarray(rng.integers(0,NODES,size=EDGES))
    x=np.ascontiguousarray(rng.normal(size=(NODES,DIM)))
    e=np.ascontiguousarray(rng.normal(size=(EDGES,DIM)))
    z=np.ascontiguousarray(rng.normal(size=(EDGES,1)))
    zh=np.ascontiguousarray(rng.normal(size=(EDGES,8,1)))
    g=dgl.graph((u,v)); g.ndata["h"]=x; g.edata["e"]=e
    ug=udgl.graph((torch.as_tensor(u),torch.as_tensor(v)))
    ug.ndata["h"]=torch.as_tensor(x); ug.edata["e"]=torch.as_tensor(e)
    return g,ug,x,e,z,zh,u,v

def seg_data():
    rng=np.random.default_rng(11)
    cuts=np.sort(rng.choice(EDGES,size=199_999,replace=False))
    seglen=np.diff(np.concatenate([[0],cuts,[EDGES]])).astype(np.int64)
    sv=np.ascontiguousarray(rng.normal(size=(EDGES,DIM)))
    return seglen,sv

def mm_data():
    rng=np.random.default_rng(12)
    a=np.ascontiguousarray(rng.normal(size=(EDGES,16)))
    b=np.ascontiguousarray(rng.normal(size=(64,16,16)))
    idx=np.ascontiguousarray(rng.integers(0,64,size=EDGES))
    return a,b,idx

def main():
    g,ug,x,e,z,zh,u,v=data()
    seglen,sv=seg_data()
    a,b,idx=mm_data()
    L=lib()
    out={}
    def rec(name, ours, theirs=None, numpy_fn=None):
        om,omed=timeit(ours)
        tm=tmed=None
        if theirs:
            tm,tmed=timeit(theirs)
        nm=nmed=None
        if numpy_fn:
            nm,nmed=timeit(numpy_fn)
        out[name]={"mojo_best_ms":om*1e3,"mojo_med_ms":omed*1e3,
                   "up_best_ms":(tm*1e3 if tm else None),"up_med_ms":(tmed*1e3 if tmed else None),
                   "np_best_ms":(nm*1e3 if nm else None),"np_med_ms":(nmed*1e3 if nmed else None)}
        print(f"{name:38s} mojo={om*1e3:8.1f}ms up={(tm*1e3 if tm else float('nan')):8.1f}ms np={(nm*1e3 if nm else float('nan')):8.1f}ms",flush=True)
        gc.collect()

    t=lambda arr: torch.as_tensor(arr)
    x4=x+4.0; e4=e+4.0
    # --- SPMM (gspmm) ---
    rec("u_mul_e_sum", lambda: ops.u_mul_e_sum(g,x,e), lambda: uops.u_mul_e_sum(ug,t(x),t(e)),
        lambda: np.add.at(np.zeros((NODES,DIM)), v, x[u]*e))
    def np_mean():
        m=np.zeros((NODES,DIM)); np.add.at(m, v, x4[u]+e4)
        d=np.bincount(v,minlength=NODES).astype(float)
        return np.where(d[:,None]>0, m/np.maximum(d,1)[:,None], 0)
    rec("u_add_e_mean", lambda: ops.u_add_e_mean(g,x4,e4), lambda: uops.u_add_e_mean(ug,t(x4),t(e4)), np_mean)
    rec("u_div_e_min", lambda: ops.u_div_e_min(g,x4,e4), lambda: uops.u_div_e_min(ug,t(x4),t(e4)))
    rec("copy_u_max", lambda: ops.copy_u_max(g,x), lambda: uops.copy_u_max(ug,t(x)),
        lambda: np.maximum.at(np.full((NODES,DIM),-np.inf), v, x[u]))
    rec("copy_e_mean", lambda: ops.copy_e_mean(g,e), lambda: uops.copy_e_mean(ug,t(e)))
    rec("u_sub_e_sum", lambda: ops.u_sub_e_sum(g,x,e), lambda: uops.u_sub_e_sum(ug,t(x),t(e)))
    rec("u_mul_e_max", lambda: ops.u_mul_e_max(g,x,e), lambda: uops.u_mul_e_max(ug,t(x),t(e)))
    # --- SDDMM (gsddmm) ---
    rec("u_dot_e", lambda: ops.u_dot_e(g,x,e), lambda: uops.u_dot_e(ug,t(x),t(e)),
        lambda: np.einsum('ij,ij->i', x[u], e))
    rec("u_mul_v", lambda: ops.u_mul_v(g,x,x), lambda: uops.u_mul_v(ug,t(x),t(x)),
        lambda: x[u]*x[v])
    rec("u_add_v", lambda: ops.u_add_v(g,x,x), lambda: uops.u_add_v(ug,t(x),t(x)),
        lambda: x[u]+x[v])
    rec("copy_u_sddmm", lambda: ops.copy_u(g,x), lambda: uops.copy_u(ug,t(x)),
        lambda: x[u].copy())
    # --- edge_softmax ---
    rec("edge_softmax_dst", lambda: ops.edge_softmax(g,z), lambda: uops.edge_softmax(ug,t(z)))
    rec("edge_softmax_src", lambda: ops.edge_softmax(g,z,norm_by="src"), lambda: uops.edge_softmax(ug,t(z),norm_by="src"))
    rec("edge_softmax_8h", lambda: ops.edge_softmax(g,zh), lambda: uops.edge_softmax(ug,t(zh)))
    # numpy edge_softmax baseline via CSR order (dst)
    indptr,indices,eid=g._in_csr()
    def np_es():
        # max per node
        m=np.full(NODES,-np.inf)
        np.maximum.at(m, v, z[:,0])
        ex=np.exp(z[:,0]-m[v])
        s=np.zeros(NODES); np.add.at(s, v, ex)
        return (ex/np.maximum(s[v],1e-300)).reshape(-1,1)
    rec("edge_softmax_dst_npcheck", lambda: ops.edge_softmax(g,z), None, np_es)
    # --- segment ---
    rec("segment_reduce_sum", lambda: ops.segment_reduce(seglen,sv,"sum"), lambda: uops.segment_reduce(torch.as_tensor(seglen),t(sv),"sum"),
        lambda: np.add.reduceat(sv, np.concatenate([[0],np.cumsum(seglen)[:-1]]), axis=0))
    rec("segment_reduce_max", lambda: ops.segment_reduce(seglen,sv,"max"), lambda: uops.segment_reduce(torch.as_tensor(seglen),t(sv),"max"),
        lambda: np.maximum.reduceat(sv, np.concatenate([[0],np.cumsum(seglen)[:-1]]), axis=0))
    rec("segment_softmax", lambda: ops.segment_softmax(seglen,sv), lambda: uops.segment_softmax(torch.as_tensor(seglen),t(sv)))
    rec("segment_mm_small", lambda: ops.segment_mm(np.ascontiguousarray(sv[:1000]), np.ascontiguousarray(np.random.default_rng(1).normal(size=(10,32,4))), np.full(10,100,dtype=np.int64)),
        lambda: uops.segment_mm(t(sv[:1000]), t(np.random.default_rng(1).normal(size=(10,32,4))), torch.full((10,),100,dtype=torch.int64)))
    # --- gather_mm ---
    rec("gather_mm", lambda: ops.gather_mm(a,b,idx_b=idx), lambda: uops.gather_mm(t(a),t(b),idx_b=torch.as_tensor(idx)),
        lambda: np.einsum('ij,ijk->ik', a, b[idx]))
    # --- update_all ---
    def ours_ua1():
        g.update_all(fn.u_mul_e("h","e","m"), fn.sum("m","z")); return g.ndata["z"]
    def theirs_ua1():
        ug.update_all(ufn.u_mul_e("h","e","m"), ufn.sum("m","z")); return ug.ndata["z"]
    rec("update_all_u_mul_e_sum", ours_ua1, theirs_ua1)
    def ours_ua2():
        g.update_all(fn.copy_u("h","m"), fn.max("m","z")); return g.ndata["z"]
    def theirs_ua2():
        ug.update_all(ufn.copy_u("h","m"), ufn.max("m","z")); return ug.ndata["z"]
    rec("update_all_copy_u_max", ours_ua2, theirs_ua2)
    # --- graph construction (one-off, not in bench) ---
    def build_g(): return dgl.graph((u,v))
    bm,bmed=timeit(build_g); print(f"graph_build mojo={bm*1e3:.1f}ms",flush=True)
    out["graph_build"]={"mojo_best_ms":bm*1e3,"mojo_med_ms":bmed*1e3}

    # ===== kernel-only vs glue =====
    print("\n=== kernel-only (direct ctypes, preallocated dst) vs public ===",flush=True)
    K={}
    indptr,indices,eid=g._in_csr()
    deg=g.in_degrees()
    # gspmm u_mul_e_sum: op MUL=2, reduce SUM=0, dl=dr=dout=32
    dst=np.empty((NODES,DIM))
    def k_gspmm_sum():
        L.mdgl_gspmm(addr(indptr),addr(indices),addr(eid),addr(x),addr(e),addr(dst),NODES,2,32,32,32,0,addr(deg))
    km,kmed=timeit(k_gspmm_sum); K["gspmm_u_mul_e_sum"]={"best_ms":km*1e3,"med_ms":kmed*1e3}
    print(f"gspmm kernel-only {km*1e3:.1f}ms",flush=True)
    # gsddmm u_mul_v: lhs u, rhs v, op MUL
    dst2=np.empty((EDGES,DIM))
    def k_sddmm():
        L.mdgl_gsddmm(addr(u),addr(v),addr(x),addr(x),addr(dst2),EDGES,0,1,2,32,32,32)
    km,kmed=timeit(k_sddmm); K["gsddmm_u_mul_v"]={"best_ms":km*1e3,"med_ms":kmed*1e3}
    print(f"gsddmm kernel-only {km*1e3:.1f}ms",flush=True)
    # gsddmm dot
    dst3=np.empty((EDGES,1))
    def k_dot():
        L.mdgl_gsddmm(addr(u),addr(v),addr(x),addr(e),addr(dst3),EDGES,0,2,4,32,32,1)
    km,kmed=timeit(k_dot); K["gsddmm_u_dot_e"]={"best_ms":km*1e3,"med_ms":kmed*1e3}
    print(f"gsddmm dot kernel-only {km*1e3:.1f}ms",flush=True)
    # edge_softmax dst d=1
    from mojodgl.ops.segment import _offsets
    dst4=np.empty((EDGES,1)); scratch=np.empty(32*2*1)
    def k_es():
        L.mdgl_edge_softmax_(addr(indptr),addr(indices),addr(eid),addr(np.ones(1,dtype=np.int8)),addr(z),addr(dst4),addr(scratch),NODES,1,1)
    km,kmed=timeit(k_es); K["edge_softmax_dst_d1"]={"best_ms":km*1e3,"med_ms":kmed*1e3}
    print(f"edge_softmax d1 kernel-only {km*1e3:.1f}ms",flush=True)
    # segment_reduce sum
    offs=_offsets(seglen); dst5=np.empty((len(seglen),DIM))
    def k_seg():
        L.mdgl_segment_reduce(addr(offs),addr(sv),addr(dst5),len(seglen),32,0,EDGES)
    km,kmed=timeit(k_seg); K["segment_reduce_sum"]={"best_ms":km*1e3,"med_ms":kmed*1e3}
    print(f"segment_reduce kernel-only {km*1e3:.1f}ms",flush=True)
    # segment_softmax
    dst6=np.zeros((EDGES,DIM)); scratch2=np.empty(32*2*32)
    def k_segsm():
        L.mdgl_segment_softmax(addr(offs),addr(sv),addr(dst6),addr(scratch2),len(seglen),32,EDGES)
    km,kmed=timeit(k_segsm); K["segment_softmax"]={"best_ms":km*1e3,"med_ms":kmed*1e3}
    print(f"segment_softmax kernel-only {km*1e3:.1f}ms",flush=True)
    # gather_mm
    dst7=np.empty((EDGES,16))
    def k_gmm():
        L.mdgl_gather_mm(addr(a),addr(b),addr(idx),addr(dst7),EDGES,16,16)
    km,kmed=timeit(k_gmm); K["gather_mm"]={"best_ms":km*1e3,"med_ms":kmed*1e3}
    print(f"gather_mm kernel-only {km*1e3:.1f}ms",flush=True)
    # cumsum / clamp / replace_inf alone (segment glue constituents)
    seg_small=seglen
    def k_cumsum():
        d=np.zeros(len(seg_small)+1,dtype=np.int64); L.mdgl_cumsum(addr(np.concatenate([[0],seg_small])),addr(d),len(seg_small)+1)
    km,_=timeit(k_cumsum); K["cumsum_200k"]={"best_ms":km*1e3}
    print(f"cumsum 200k {km*1e3:.3f}ms",flush=True)

    # ===== ctypes call count per public call =====
    print("\n=== ctypes calls per public call ===",flush=True)
    counts={}
    real_lib=None
    import importlib
    segmod=importlib.import_module('mojodgl.ops.segment'); esmod=importlib.import_module('mojodgl.ops.edge_softmax'); gmmod=importlib.import_module('mojodgl.ops.gather_mm')
    spmod=importlib.import_module('mojodgl.ops.spmm'); sddmod=importlib.import_module('mojodgl.ops.sddmm')
    # wrap lib() to return proxy
    class Proxy:
        def __init__(self, real): self._r=real; self.n=0
        def __getattr__(self, name):
            fnc=getattr(self._r,name)
            if not name.startswith("mdgl_"): return fnc
            def w(*a,**k):
                self.n+=1; return fnc(*a,**k)
            return w
    proxy=Proxy(L)
    orig={}
    for m in (segmod,esmod,gmmod,spmod,sddmod):
        orig[m]=vars(m)['lib']
    for m in orig:
        def make(p=proxy):
            return lambda: p
        vars(m)['lib']=make()
    def count(name, fn_):
        proxy.n=0; fn_(); counts[name]=proxy.n; print(f"{name}: {proxy.n} ctypes calls",flush=True)
    count("u_mul_e_sum", lambda: ops.u_mul_e_sum(g,x,e))
    count("u_mul_v", lambda: ops.u_mul_v(g,x,x))
    count("copy_u_max", lambda: ops.copy_u_max(g,x))
    count("edge_softmax_dst", lambda: ops.edge_softmax(g,z))
    count("segment_reduce_sum", lambda: ops.segment_reduce(seglen,sv,"sum"))
    count("segment_reduce_mean", lambda: ops.segment_reduce(seglen,sv,"mean"))
    count("segment_reduce_max", lambda: ops.segment_reduce(seglen,sv,"max"))
    count("segment_softmax", lambda: ops.segment_softmax(seglen,sv))
    count("gather_mm_fused", lambda: ops.gather_mm(a,b,idx_b=idx))
    # gather_mm sorted route: big block
    a2=np.ascontiguousarray(np.random.default_rng(2).normal(size=(10000,256)))
    b2=np.ascontiguousarray(np.random.default_rng(3).normal(size=(8,256,256)))
    idx2=np.ascontiguousarray(np.random.default_rng(4).integers(0,8,size=10000))
    count("gather_mm_sorted_route_256", lambda: ops.gather_mm(a2,b2,idx_b=idx2))
    count("update_all_u_mul_e_sum", ours_ua1)
    count("edge_softmax_eids_subset", lambda: ops.edge_softmax(g,z,eids=np.arange(0,EDGES,2,dtype=np.int64)))
    for m,f in orig.items(): vars(m)['lib']=f

    # ===== allocation bytes per public call (numpy alloc counting) =====
    print("\n=== numpy allocations per public call ===",flush=True)
    alloc={}
    real_empty, real_zeros, real_repeat = np.empty, np.zeros, np.repeat
    stats={"bytes":0,"calls":0}
    def wrap(orig_):
        def w(shape,*a,**k):
            r=orig_(shape,*a,**k)
            try: stats["bytes"]+=r.nbytes; stats["calls"]+=1
            except Exception: pass
            return r
        return w
    def measure(name, fn_):
        stats["bytes"]=0; stats["calls"]=0
        np.empty, np.zeros, np.repeat = wrap(real_empty), wrap(real_zeros), wrap(real_repeat)
        try: fn_()
        finally: np.empty, np.zeros, np.repeat = real_empty, real_zeros, real_repeat
        alloc[name]={"calls":stats["calls"],"bytes":stats["bytes"]}
        print(f"{name}: {stats['calls']} np allocs, {stats['bytes']/1e6:.2f} MB",flush=True)
    measure("u_mul_e_sum", lambda: ops.u_mul_e_sum(g,x,e))
    measure("u_mul_v", lambda: ops.u_mul_v(g,x,x))
    measure("copy_u_max", lambda: ops.copy_u_max(g,x))
    measure("edge_softmax_dst", lambda: ops.edge_softmax(g,z))
    measure("edge_softmax_8h", lambda: ops.edge_softmax(g,zh))
    measure("segment_reduce_sum", lambda: ops.segment_reduce(seglen,sv,"sum"))
    measure("segment_reduce_mean", lambda: ops.segment_reduce(seglen,sv,"mean"))
    measure("segment_reduce_max", lambda: ops.segment_reduce(seglen,sv,"max"))
    measure("segment_softmax", lambda: ops.segment_softmax(seglen,sv))
    measure("gather_mm_fused", lambda: ops.gather_mm(a,b,idx_b=idx))
    measure("update_all_u_mul_e_sum", ours_ua1)
    measure("update_all_copy_u_max", ours_ua2)
    measure("edge_softmax_eids_half", lambda: ops.edge_softmax(g,z,eids=np.arange(0,EDGES,2,dtype=np.int64)))

    # ===== small-size scaling (fixed overhead) =====
    print("\n=== small-size scaling ===",flush=True)
    small={}
    for E_,tag in [(1000,"1k"),(10000,"10k"),(100000,"100k")]:
        rng=np.random.default_rng(7)
        uu=np.ascontiguousarray(rng.integers(0,NODES,size=E_)); vv=np.ascontiguousarray(rng.integers(0,NODES,size=E_))
        gg=dgl.graph((uu,vv))
        xx=np.ascontiguousarray(rng.normal(size=(gg.number_of_nodes(),8))); ee=np.ascontiguousarray(rng.normal(size=(E_,8)))
        # use subset nodes to keep x small? keep full NODES for realism of gspmm dst alloc
        bm,_=timeit(lambda: ops.u_mul_e_sum(gg,xx,ee),repeat=5)
        small[f"gspmm_{tag}"]={"best_ms":bm*1e3}
        print(f"gspmm E={E_} (N=200k,8d): {bm*1e3:.2f}ms",flush=True)
        bm2,_=timeit(lambda: ops.u_mul_v(gg,xx,xx),repeat=5)
        small[f"gsddmm_{tag}"]={"best_ms":bm2*1e3}
        print(f"gsddmm E={E_}: {bm2*1e3:.2f}ms",flush=True)
        # tiny graph both axes small
        NN=1000
        uu2=np.ascontiguousarray(rng.integers(0,NN,size=E_)); vv2=np.ascontiguousarray(rng.integers(0,NN,size=E_))
        gg2=dgl.graph((uu2,vv2)); xx2=np.ascontiguousarray(rng.normal(size=(NN,8)))
        bm3,_=timeit(lambda: ops.u_mul_e_sum(gg2,xx2,ee),repeat=5)
        small[f"gspmm_smallN_{tag}"]={"best_ms":bm3*1e3}
        print(f"gspmm smallN E={E_}: {bm3*1e3:.2f}ms",flush=True)
    # ctypes raw overhead: empty kernel on tiny arrays
    tiny=np.zeros(10); tinyd=np.zeros(10,dtype=np.int64)
    def k_tiny(): L.mdgl_cumsum(addr(tinyd),addr(tinyd),10)
    tm,_=timeit(k_tiny,repeat=1000); print(f"raw ctypes tiny cumsum: {tm*1e6:.2f}us",flush=True)
    small["raw_ctypes_us"]=tm*1e6

    print("\n=== thread scaling check: upstream torch threads ===")
    print("torch threads", torch.get_num_threads())

    with open("/tmp/profile_deep.json","w") as f:
        json.dump({"public":out,"kernel":K,"ctypes_counts":counts,"alloc":alloc,"small":small},f,indent=2)
    print("wrote /tmp/profile_deep.json")

if __name__=="__main__":
    main()
