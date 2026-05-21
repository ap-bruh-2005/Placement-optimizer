# placer.py — PI-GNN hybrid analytical placer submission
# Generated from Updated2.ipynb inference code.
import math
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except Exception as exc:
    Data = None
    GCNConv = None
    global_mean_pool = None
    HAS_PYG = False
    _PYG_IMPORT_ERROR = exc
    print("Import path skipped")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_valid_proxy_cost(costs):
    if not isinstance(costs, dict):
        return False
    if "proxy_error" in costs or "proxy_cost" not in costs:
        return False
    try:
        proxy = float(costs["proxy_cost"])
        if not math.isfinite(proxy):
            return False
    except Exception:
        return False
    try:
        if int(costs.get("overlap_count", 0)) != 0:
            return False
    except Exception:
        return False
    return True


class MyPlacer:
    """
    It runs an analytical candidate and, if the packaged PyG model/weights load,
    a PI-GNN hybrid candidate. It returns the best valid official proxy candidate.
    """
    def __init__(self):
        self.device = device
        self.model = None
        self.model_ok = False
        self.plc_cache = {}

        if HAS_PYG:
            try:
                weights_path = Path(__file__).resolve().parent / "pi_gnn_weights.pt"
                self.model = PICongesionGNN(in_dim=17, hidden_dim=64).to(self.device)
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.eval()
                self.model_ok = True
                print(f"[MyPlacer] loaded GNN weights from {weights_path.name}")
            except Exception as exc:
                print(f"[MyPlacer] WARNING: GNN disabled; failed to load weights: {exc}")
                self.model = None
                self.model_ok = False
        else:
            print(f"[MyPlacer] WARNING: PyG unavailable; using analytical fallback only: {_PYG_IMPORT_ERROR}")

    def place(self, benchmark):
        plc = self._get_plc(benchmark)
        candidates = []

        # Analytical fallback candidate. Keep this first because it is the safest.
        try:
            ana = AnalyticalPlacer(
                steps=120,
                adam_steps=120,
                lr=0.020,
                lambda_dens_end=3.0,
                lambda_repel=1000.0,
                seeds=(0, 1),
                init_jitter=0.003,
                official_eval_every=10,
                checkpoint_legalize=True,
                checkpoint_verbose=False,
                adaptive_lr=False,
            )
            pos_ana = ana.place(benchmark, plc)
            cost_ana = compute_proxy_cost(pos_ana, benchmark, plc)
            if is_valid_proxy_cost(cost_ana):
                candidates.append(("analytical", float(cost_ana["proxy_cost"]), pos_ana))
                print(f"[MyPlacer] analytical proxy={float(cost_ana['proxy_cost']):.6f}")
            else:
                print(f"[MyPlacer] analytical invalid/bad cost: {cost_ana}")
        except Exception as exc:
            print(f"[MyPlacer] analytical failed: {type(exc).__name__}: {exc}")

        # GNN hybrid candidate. It is only accepted if valid and lower proxy.
        if self.model_ok:
            try:
                gnn = HybridAnalyticalPlacer(
                    self.model,
                    steps=160,
                    adam_steps=160,
                    lbfgs_steps=0,
                    lbfgs_lr=0.0,
                    lr=0.020,
                    lambda_dens_end=3.0,
                    lambda_cong_start=0.0,
                    lambda_cong_end=0.10,
                    lambda_repel=500.0,
                    seeds=(0,),
                    use_softplus_cong=False,
                    init_jitter=0.003,
                    official_eval_every=10,
                    checkpoint_legalize=True,
                    checkpoint_verbose=False,
                    adaptive_lr=False,
                )
                pos_gnn = gnn.place(benchmark, plc)
                cost_gnn = compute_proxy_cost(pos_gnn, benchmark, plc)
                if is_valid_proxy_cost(cost_gnn):
                    candidates.append(("gnn_hybrid", float(cost_gnn["proxy_cost"]), pos_gnn))
                    print(f"[MyPlacer] gnn_hybrid proxy={float(cost_gnn['proxy_cost']):.6f}")
                else:
                    print(f"[MyPlacer] gnn_hybrid invalid/bad cost: {cost_gnn}")
            except Exception as exc:
                print(f"[MyPlacer] GNN path failed: {type(exc).__name__}: {exc}")

        if candidates:
            name, proxy, pos = min(candidates, key=lambda x: x[1])
            print(f"[MyPlacer] selected {name} proxy={proxy:.6f}")
            return pos.detach().cpu()

        # Last-resort validity fallback. This should be rare.
        print("[MyPlacer] WARNING: no valid optimized candidate; returning legalized initial placement")
        return legalize_hard_macros(benchmark.macro_positions.clone(), benchmark, gap=1e-3).detach().cpu()

    def _get_plc(self, benchmark):
        name = getattr(benchmark, "name", None)
        if name is None:
            raise RuntimeError("Benchmark has no name attribute; cannot locate .plc")

        if name in self.plc_cache:
            return self.plc_cache[name]

        # Tier 1 IBM benchmarks.
        ibm_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        if ibm_dir.exists():
            _, plc = load_benchmark_from_dir(str(ibm_dir))
            self.plc_cache[name] = plc
            return plc

        # Optional public NG45 paths. Only used if local files exist.
        ng45_roots = [
            Path("external/MacroPlacement/Flows/NanGate45") / name / "netlist/output_CT_Grouping",
            Path("external/MacroPlacement/Flows/NanGate45") / name,
        ]
        for root in ng45_roots:
            if root.exists():
                # Try the repo loader style if available.
                from macro_place.loader import load_benchmark
                netlist = root / "netlist.pb.txt"
                plc_file = root / "initial.plc"
                if netlist.exists() and plc_file.exists():
                    _, plc = load_benchmark(str(netlist), str(plc_file), name=name)
                    self.plc_cache[name] = plc
                    return plc

        raise RuntimeError(f"Could not locate benchmark directory/.plc for {name}")



def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Geometry helpers
# =============================================================================

def clamp_centers_to_canvas(pos, sizes, canvas_w, canvas_h, gap=1e-3):
    half = sizes / 2.0
    x = pos[:,0].clamp(min=half[:,0]+gap, max=canvas_w-half[:,0]-gap)
    y = pos[:,1].clamp(min=half[:,1]+gap, max=canvas_h-half[:,1]-gap)
    return torch.stack([x, y], dim=1)

def rect_bounds(pos, sizes):
    half = sizes / 2.0
    return pos[:,0]-half[:,0], pos[:,0]+half[:,0], pos[:,1]-half[:,1], pos[:,1]+half[:,1]

def pairwise_overlap_matrix(pos, sizes, gap=1e-3):
    l,r,b,t = rect_bounds(pos, sizes)
    l=l-gap/2; r=r+gap/2; b=b-gap/2; t=t+gap/2
    ox = torch.clamp(torch.minimum(r[:,None],r[None,:])-torch.maximum(l[:,None],l[None,:]),min=0.)
    oy = torch.clamp(torch.minimum(t[:,None],t[None,:])-torch.maximum(b[:,None],b[None,:]),min=0.)
    eye = torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
    ox[eye]=0.; oy[eye]=0.
    return ox, oy

def hard_overlap_count(pos, benchmark, gap=1e-3):
    idx = torch.where(benchmark.get_hard_macro_mask())[0]
    if len(idx) <= 1: return 0
    ox,oy = pairwise_overlap_matrix(pos[idx], benchmark.macro_sizes[idx], gap=gap)
    return int(((ox>0)&(oy>0)).sum().item()//2)



# newer version: Final-only full pairwise legalization replaces the grid-neighbor version above.
# newer version: This is intentionally O(N^2), but it must be called only after optimization, not inside the loop.
def legalize_hard_macros(pos, benchmark, gap=1e-3, max_passes=200, damping=1.0):
    pos = pos.detach().clone().cpu()
    sizes = benchmark.macro_sizes.detach().clone().cpu()
    fixed = benchmark.macro_fixed.detach().clone().cpu().bool()
    hard = benchmark.get_hard_macro_mask().detach().clone().cpu().bool()
    movable = benchmark.get_movable_mask().detach().clone().cpu().bool()
    original = benchmark.macro_positions.detach().clone().cpu()

    W = float(benchmark.canvas_width)
    H = float(benchmark.canvas_height)
    pos = clamp_centers_to_canvas(pos, sizes, W, H, gap)
    pos[~movable] = original[~movable]

    hard_idx = torch.where(hard)[0]
    if len(hard_idx) <= 1:
        return pos

    for _pass in range(max_passes):
        ph = pos[hard_idx]
        sh = sizes[hard_idx]
        ox, oy = pairwise_overlap_matrix(ph, sh, gap=gap)
        overlap = ox * oy
        pair_mask = torch.triu((ox > 0) & (oy > 0), diagonal=1)
        pairs = pair_mask.nonzero(as_tuple=False)
        if pairs.numel() == 0:
            break

        pair_area = overlap[pairs[:, 0], pairs[:, 1]]
        order = torch.argsort(pair_area, descending=True)
        changed = False
        touched = set()

        for ord_idx in order.tolist():
            a = int(pairs[ord_idx, 0])
            b = int(pairs[ord_idx, 1])
            i = int(hard_idx[a])
            j = int(hard_idx[b])
            if i in touched or j in touched:
                continue

            li = pos[i, 0] - sizes[i, 0] / 2.0 - gap / 2.0
            ri = pos[i, 0] + sizes[i, 0] / 2.0 + gap / 2.0
            bi = pos[i, 1] - sizes[i, 1] / 2.0 - gap / 2.0
            ti = pos[i, 1] + sizes[i, 1] / 2.0 + gap / 2.0
            lj = pos[j, 0] - sizes[j, 0] / 2.0 - gap / 2.0
            rj = pos[j, 0] + sizes[j, 0] / 2.0 + gap / 2.0
            bj = pos[j, 1] - sizes[j, 1] / 2.0 - gap / 2.0
            tj = pos[j, 1] + sizes[j, 1] / 2.0 + gap / 2.0

            ox2 = min(ri, rj) - max(li, lj)
            oy2 = min(ti, tj) - max(bi, bj)
            if ox2 <= 0 or oy2 <= 0:
                continue

            dx = float(pos[j, 0] - pos[i, 0])
            dy = float(pos[j, 1] - pos[i, 1])
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                dx = 1.0 if j > i else -1.0
                dy = 0.0

            if ox2 <= oy2:
                mv = torch.tensor([(float(ox2) + gap) * (1.0 if dx >= 0 else -1.0), 0.0], dtype=pos.dtype)
            else:
                mv = torch.tensor([0.0, (float(oy2) + gap) * (1.0 if dy >= 0 else -1.0)], dtype=pos.dtype)
            mv = damping * mv

            i_mov = bool(movable[i] and not fixed[i])
            j_mov = bool(movable[j] and not fixed[j])
            if i_mov and j_mov:
                pos[i] -= 0.5 * mv
                pos[j] += 0.5 * mv
                touched.add(i)
                touched.add(j)
                changed = True
            elif i_mov:
                pos[i] -= mv
                touched.add(i)
                changed = True
            elif j_mov:
                pos[j] += mv
                touched.add(j)
                changed = True

            if changed:
                pos = clamp_centers_to_canvas(pos, sizes, W, H, gap)
                pos[~movable] = original[~movable]

        if not changed:
            break

    pos = clamp_centers_to_canvas(pos, sizes, W, H, gap)
    pos[~movable] = original[~movable]
    return pos


# =============================================================================
# Net extraction from PlacementCost (plc)
#
# Net data lives in plc.nets, NOT benchmark tensors.
# plc.nets : {driver_pin -> [sink_pins]}, pin names = "MACRO/PIN"
# Results cached per benchmark — computed once per session.
# =============================================================================

_NET_CACHE  = {}
_EDGE_CACHE = {}
_CONN_CACHE = {}
_HPWL_CACHE = {}

def _get_nets_from_plc(plc, benchmark, max_fanout=20):
    key = benchmark.name
    if key in _NET_CACHE: return _NET_CACHE[key]

    macro_name_to_tensor = {}
    for t_idx, plc_idx in enumerate(benchmark.hard_macro_indices):
        try:
            macro_name_to_tensor[plc.modules_w_pins[plc_idx].get_name()] = t_idx
        except Exception: pass
    for t_off, plc_idx in enumerate(benchmark.soft_macro_indices):
        try:
            macro_name_to_tensor[plc.modules_w_pins[plc_idx].get_name()] =                 benchmark.num_hard_macros + t_off
        except Exception: pass

    nets = []
    for driver_pin, sink_pins in plc.nets.items():
        macro_set = set()
        for pin in [driver_pin] + list(sink_pins):
            if '/' in pin:
                name = pin.rsplit('/',1)[0]
                if name in macro_name_to_tensor:
                    macro_set.add(macro_name_to_tensor[name])
        if 2 <= len(macro_set) <= max_fanout:
            nets.append(torch.tensor(sorted(macro_set), dtype=torch.long))

    _NET_CACHE[key] = nets
    print(f"  [{benchmark.name}] {len(nets)} nets extracted from plc")
    return nets


def build_net_edges(plc, benchmark, k_fallback=8):
    key = benchmark.name
    if key in _EDGE_CACHE: return _EDGE_CACHE[key]
    nets = _get_nets_from_plc(plc, benchmark)
    if not nets:
        ei = build_knn_edges(benchmark.macro_positions, k=k_fallback)
        _EDGE_CACHE[key] = ei; return ei
    edge_set = set()
    for net in nets:
        idx = net.tolist()
        for a in range(len(idx)):
            for b in range(a+1,len(idx)):
                edge_set.add((idx[a],idx[b])); edge_set.add((idx[b],idx[a]))
    ei = torch.tensor(list(edge_set),dtype=torch.long).t().contiguous() if edge_set          else build_knn_edges(benchmark.macro_positions, k=k_fallback)
    _EDGE_CACHE[key] = ei; return ei


def build_knn_edges(pos, k=8):
    n = pos.shape[0]
    if n <= 1: return torch.zeros((2,0),dtype=torch.long)
    k = min(k, n-1)
    dist = torch.cdist(pos,pos,p=2); dist.fill_diagonal_(float("inf"))
    knn  = dist.topk(k=k,largest=False).indices
    edges = [(i,j) for i in range(n) for j in knn[i].tolist()] +             [(j,i) for i in range(n) for j in knn[i].tolist()]
    return torch.tensor(edges,dtype=torch.long).t().contiguous()


def build_net_plus_geom_edges(plc, benchmark, pos, k_net_fallback=8, k_geom=8):
    """
    Build the graph used by the GNN as a union of:
      1. static netlist connectivity edges
      2. dynamic geometric KNN edges from the current placement

    Why this patch matters:
      netlist-only edges make it easy for the GNN to learn a benchmark-level
      average congestion value. Congestion is spatial, so the model also needs
      local geometry edges that change with placement.

    Note:
      edge construction itself is non-differentiable because it uses KNN/top-k,
      but the node features still depend on P, so gradients can flow through
      make_node_features(...).
    """
    net_ei = build_net_edges(plc, benchmark, k_fallback=k_net_fallback).detach().cpu()

    pos_cpu = pos.detach().cpu()
    geom_ei = build_knn_edges(pos_cpu, k=k_geom).detach().cpu()

    if net_ei.numel() == 0:
        ei = geom_ei
    elif geom_ei.numel() == 0:
        ei = net_ei
    else:
        ei = torch.cat([net_ei, geom_ei], dim=1)

    if ei.numel() == 0:
        return ei.long().contiguous()

    # Deduplicate directed edges.
    ei = torch.unique(ei.t().long(), dim=0).t().contiguous()
    return ei


def _get_conn_features(plc, benchmark):
    key = benchmark.name
    if key in _CONN_CACHE: return _CONN_CACHE[key]
    nets = _get_nets_from_plc(plc, benchmark)
    N = benchmark.num_macros
    nd=torch.zeros(N); nw=torch.zeros(N); mf=torch.zeros(N)
    for net in nets:
        f=float(net.numel())
        for i in net.tolist():
            if i<N:
                nd[i]+=1.; nw[i]+=f
                if f>mf[i].item(): mf[i]=f
    conn = torch.stack([torch.log1p(nd),torch.log1p(nw),torch.log1p(mf)],dim=1)
    _CONN_CACHE[key] = conn; return conn


def make_node_features(benchmark, plc, pos):
    # Ensure benchmark tensors are on the same device as pos
    sizes=benchmark.macro_sizes.to(pos.device)
    w,h=sizes[:,0:1],sizes[:,1:2]
    area=w*h; aspect=w/(h+1e-8)
    W=float(benchmark.canvas_width); H=float(benchmark.canvas_height)
    N=float(benchmark.num_macros)
    x_n=pos[:,0:1]/W; y_n=pos[:,1:2]/H
    half=sizes/2.0
    left=(pos[:,0:1]-half[:,0:1])/W
    right=(W-(pos[:,0:1]+half[:,0:1]))/W
    bottom=(pos[:,1:2]-half[:,1:2])/H
    top_m=(H-(pos[:,1:2]+half[:,1:2]))/H
    is_hard=benchmark.get_hard_macro_mask().float().unsqueeze(1).to(pos.device)
    is_soft=benchmark.get_soft_macro_mask().float().unsqueeze(1).to(pos.device)
    is_fixed=benchmark.macro_fixed.float().unsqueeze(1).to(pos.device)
    n_feat=torch.full((benchmark.num_macros,1),math.log1p(N),dtype=torch.float32,device=pos.device)
    conn=_get_conn_features(plc,benchmark).to(pos.device)
    return torch.cat([torch.log1p(w),torch.log1p(h),torch.log1p(area),aspect,
                      x_n,y_n,left,right,bottom,top_m,
                      is_hard,is_soft,is_fixed,n_feat,conn],dim=1)


# =============================================================================
# Physics labels  (vectorised, GPU-friendly)
# =============================================================================

def _get_hpwl_index(plc, benchmark):
    key = benchmark.name
    if key in _HPWL_CACHE: return _HPWL_CACHE[key]
    nets = _get_nets_from_plc(plc, benchmark)
    ni, mi = [], []
    for n_idx, net in enumerate(nets):
        for m in net.tolist(): ni.append(n_idx); mi.append(m)
    result = (torch.tensor(ni,dtype=torch.long),
              torch.tensor(mi,dtype=torch.long), len(nets))
    _HPWL_CACHE[key] = result; return result

def smooth_hpwl(P, benchmark, plc, alpha=0.5):
    net_idx, macro_idx, num_nets = _get_hpwl_index(plc, benchmark)
    if num_nets == 0:
        return torch.tensor(0., device=P.device, dtype=P.dtype)
    net_idx   = net_idx.to(P.device)
    macro_idx = macro_idx.to(P.device)
    coords = P[macro_idx]
    px,py  = coords[:,0]/alpha, coords[:,1]/alpha

    def lse(vals):
        vmax = torch.zeros(num_nets,device=P.device,dtype=P.dtype)
        vmax.scatter_reduce_(0,net_idx,vals,reduce="amax",include_self=True)
        esum = torch.zeros(num_nets,device=P.device,dtype=P.dtype)
        esum.scatter_add_(0,net_idx,torch.exp(vals-vmax[net_idx]))
        return alpha*(vmax+torch.log(esum+1e-12))

    return (lse(px)+lse(-px)+lse(py)+lse(-py)).sum()

def density_penalty(P, benchmark, bins=16):
    W=float(benchmark.canvas_width); H=float(benchmark.canvas_height)
    sizes=benchmark.macro_sizes.to(P.device)
    areas=(sizes[:,0]*sizes[:,1]).to(P.dtype)
    cx=(P[:,0]/W*bins).clamp(1e-6,bins-1e-6)
    cy=(P[:,1]/H*bins).clamp(1e-6,bins-1e-6)
    x0=cx.floor().long().clamp(0,bins-1); x1=(x0+1).clamp(0,bins-1)
    y0=cy.floor().long().clamp(0,bins-1); y1=(y0+1).clamp(0,bins-1)
    wx1=cx-cx.floor(); wy1=cy-cy.floor(); wx0=1.-wx1; wy0=1.-wy1
    density=torch.zeros(bins*bins,dtype=P.dtype,device=P.device)
    def deposit(bx,by,w): density.scatter_add_(0,by*bins+bx,areas*w)
    deposit(x0,y0,wx0*wy0); deposit(x1,y0,wx1*wy0)
    deposit(x0,y1,wx0*wy1); deposit(x1,y1,wx1*wy1)
    target=areas.sum()/(bins*bins)
    return (torch.relu(density-target)**2).mean()/(target**2+1e-8)

def compute_physics_labels(pos, benchmark, plc, hpwl_alpha=0.5, density_bins=16):
    W=float(benchmark.canvas_width); H=float(benchmark.canvas_height)
    hpwl_norm=smooth_hpwl(pos,benchmark,plc,alpha=hpwl_alpha)/(W+H)
    dens_norm =density_penalty(pos,benchmark,bins=density_bins)
    return hpwl_norm.item(), dens_norm.item()


# =============================================================================
# Graph sample
# =============================================================================

def make_graph_sample(benchmark, plc, placement, costs, benchmark_name,
                      k_fallback=8, k_geom=8, target_key="congestion_cost",
                      hpwl_alpha=0.5, density_bins=16):
    x          = make_node_features(benchmark, plc, placement)
    edge_index = build_net_plus_geom_edges(
        plc, benchmark, placement,
        k_net_fallback=k_fallback,
        k_geom=k_geom,
    )
    y_cong     = torch.tensor([float(costs[target_key])], dtype=torch.float32)
    hv, dv     = compute_physics_labels(placement, benchmark, plc, hpwl_alpha, density_bins)

    data = Data(x=x, edge_index=edge_index,
                y=y_cong,
                y_wl  =torch.tensor([hv], dtype=torch.float32),
                y_dens=torch.tensor([dv], dtype=torch.float32),
                pos=placement.clone())
    data.benchmark_name  = benchmark_name
    data.proxy_cost      = float(costs["proxy_cost"])
    data.wirelength_cost = float(costs["wirelength_cost"])
    data.density_cost    = float(costs["density_cost"])
    data.congestion_cost = float(costs["congestion_cost"])
    data.overlap_count   = int(costs["overlap_count"])
    return data

def boundary_penalty(P, benchmark, gap=1e-3):
    sizes = benchmark.macro_sizes.to(P.device)
    W, H = float(benchmark.canvas_width), float(benchmark.canvas_height)
    half = sizes / 2.0
    return (
        F.relu(half[:, 0] + gap - P[:, 0])
        + F.relu(P[:, 0] - (W - half[:, 0] - gap))
        + F.relu(half[:, 1] + gap - P[:, 1])
        + F.relu(P[:, 1] - (H - half[:, 1] - gap))
    ).mean()

def _safe_scale(x, floor=1e-3, default=1.0):
    try:
        if torch.is_tensor(x):
            v = float(x.detach().cpu())
        else:
            v = float(x)

        v = abs(v)

        if not math.isfinite(v) or v < floor:
            return torch.tensor(default, device=x.device if torch.is_tensor(x) else device)

        return torch.tensor(max(v, floor), device=x.device if torch.is_tensor(x) else device)

    except Exception:
        return torch.tensor(default, device=device)

def _objective_scales(P0, benchmark, plc, trained_model=None, k_graph=8):
    with torch.no_grad():
        W, H = float(benchmark.canvas_width), float(benchmark.canvas_height)
        wl0 = smooth_hpwl(P0, benchmark, plc) / (W + H)
        dens0 = density_penalty(P0, benchmark)
        scales = {
            "wl": _safe_scale(wl0),
            "dens": _safe_scale(dens0),
            "cong": torch.tensor(1.0, device=P0.device),
        }
        if trained_model is not None:
            cong0 = gnn_congestion_term(P0, benchmark, plc, trained_model, k_graph=k_graph)
            scales["cong"] = _safe_scale(cong0)
    return scales

# newer version: Full O(N^2) hard macro overlap penalty.
# newer version: Use this in both analytical and hybrid if runtime is acceptable.
# newer version: This is differentiable and safe inside Adam; it is NOT the hard legalizer.
def full_hard_repulsion_penalty(P, benchmark, gap=1e-3):
    hard = benchmark.get_hard_macro_mask().to(P.device)
    idx = torch.where(hard)[0]

    if idx.numel() <= 1:
        return torch.tensor(0.0, device=P.device, dtype=P.dtype)

    Ph = P[idx]
    sizes = benchmark.macro_sizes.to(P.device)[idx]
    half = sizes / 2.0

    left = Ph[:, 0] - half[:, 0] - gap / 2.0
    right = Ph[:, 0] + half[:, 0] + gap / 2.0
    bottom = Ph[:, 1] - half[:, 1] - gap / 2.0
    top = Ph[:, 1] + half[:, 1] + gap / 2.0

    ox = torch.minimum(right[:, None], right[None, :]) - torch.maximum(left[:, None], left[None, :])
    oy = torch.minimum(top[:, None], top[None, :]) - torch.maximum(bottom[:, None], bottom[None, :])

    overlap = torch.clamp(ox, min=0.0) * torch.clamp(oy, min=0.0)

    # newer version: remove diagonal self-overlap.
    n = idx.numel()
    overlap = overlap * (1.0 - torch.eye(n, device=P.device, dtype=P.dtype))

    # newer version: each pair appears twice.
    overlap_area = overlap.sum() / 2.0
    total_hard_area = (sizes[:, 0] * sizes[:, 1]).sum().clamp_min(1e-8)

    return overlap_area / total_hard_area


# newer version: GNN congestion term.
# newer version: Default is raw prediction, because softplus can flatten/shift the gradient.
# newer version: If raw outputs become unstable, set use_softplus=True.
def gnn_congestion_term(P, benchmark, plc, trained_model, k_graph=8, k_geom=8, use_softplus=False):
    trained_model.eval()

    x = make_node_features(benchmark, plc, P).to(P.device)
    ei = build_net_plus_geom_edges(
        plc, benchmark, P,
        k_net_fallback=k_graph,
        k_geom=k_geom,
    ).to(P.device)

    data = Data(
        x=x,
        edge_index=ei,
        batch=torch.zeros(P.shape[0], dtype=torch.long, device=P.device),
    )

    # Supports congestion-only models and older tuple/list outputs.
    out = trained_model(data)
    pred_cong = out[0] if isinstance(out, (tuple, list)) else out

    if use_softplus:
        return F.softplus(pred_cong).mean()

    return pred_cong.mean()


# newer version: Backward-compatible wrapper.
# newer version: Congestion is the only learned term used by the hybrid placer.
def gnn_physics_terms(P, benchmark, plc, trained_model, k_graph=8):
    J_cong = gnn_congestion_term(
        P,
        benchmark,
        plc,
        trained_model,
        k_graph=k_graph,
        use_softplus=False,
    )
    W, H = float(benchmark.canvas_width), float(benchmark.canvas_height)
    J_wl = smooth_hpwl(P, benchmark, plc) / (W + H)
    J_dens = density_penalty(P, benchmark)
    return J_cong, J_wl, J_dens


# newer version: Hybrid adaptive weights.
# newer version: Congestion can be strong, but only after overlap pressure is basically gone.
def adaptive_hybrid_weights(
    progress,
    J_repel,
    base_lam_d,
    base_lam_c,
    base_lam_r,
    repel_lo=1e-6,
    repel_hi=1e-4,
    max_lam_r=20000.0,
):
    r = float(J_repel.detach().cpu())

    # newer version: density follows normal annealing.
    lam_d_eff = base_lam_d

    if r > repel_hi:
        # newer version: Overlap still meaningful.
        # newer version: No GNN steering. Make legality dominant.
        lam_c_eff = 0.0
        lam_r_eff = min(max_lam_r, base_lam_r * 10.0 * (1.0 + 2.0 * progress))

    elif r > repel_lo:
        # newer version: Nearly legal.
        # newer version: Weakly turn on GNN while still prioritizing repulsion.
        lam_c_eff = 0.25 * base_lam_c
        lam_r_eff = min(max_lam_r, base_lam_r * 4.0 * (1.0 + progress))

    else:
        # newer version: Legal-ish geometry.
        # newer version: Now the GNN congestion residual is allowed to matter.
        lam_c_eff = base_lam_c
        lam_r_eff = base_lam_r

    return lam_d_eff, lam_c_eff, lam_r_eff


# newer version: Snapshot score prioritizes validity pressure first.
def legality_first_score(loss, J_boundary, J_repel):
    return (
        1e7 * float(J_repel.detach().cpu())
        + 1e4 * float(J_boundary.detach().cpu())
        + float(loss.detach().cpu())
    )

def adaptive_analytical_weights(
    progress,
    J_repel,
    base_lam_d,
    base_lam_r,
    repel_lo=1e-6,
    repel_hi=1e-4,
    max_lam_r=20000.0,
):
    r = float(J_repel.detach().cpu())

    if r > repel_hi:
        lam_d_eff = 0.25 * base_lam_d
        lam_r_eff = min(max_lam_r, base_lam_r * 10.0 * (1.0 + 2.0 * progress))
    elif r > repel_lo:
        lam_d_eff = 0.75 * base_lam_d
        lam_r_eff = min(max_lam_r, base_lam_r * 4.0 * (1.0 + progress))
    else:
        lam_d_eff = base_lam_d
        lam_r_eff = base_lam_r

    return lam_d_eff, lam_r_eff

def smooth_late_ramp_progress(progress, start_frac=0.45):
    """
    0 before start_frac, then smooth ramp to 1.
    progress should be in [0, 1].
    """
    if progress <= start_frac:
        return 0.0

    t = (progress - start_frac) / max(1e-8, 1.0 - start_frac)
    t = max(0.0, min(1.0, t))

    # smoothstep
    return t * t * (3.0 - 2.0 * t)


def _safe_grad_norm(loss, var, eps=1e-12):
    """
    Computes ||d loss / d var|| while preserving the graph for later backward().
    """
    if not torch.is_grad_enabled():
        return torch.tensor(0.0, device=var.device)

    g = torch.autograd.grad(
        loss,
        var,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]

    if g is None:
        return torch.tensor(0.0, device=var.device)

    return g.detach().norm().clamp_min(eps)


def adaptive_gnn_lambda_from_grad(
    backbone_loss,
    gnn_loss,
    opt_var,
    target_ratio=0.20,
    max_lambda=0.25,
    eps=1e-12,
):
    """
    Pick lambda so that:

        ||lambda * grad_gnn|| ~= target_ratio * ||grad_backbone||

    This is better than blindly increasing lambda_cong_end.
    """
    if not torch.is_grad_enabled():
        return 0.0, {
            "grad_backbone": 0.0,
            "grad_gnn": 0.0,
            "lambda_adapt": 0.0,
            "grad_ratio": 0.0,
        }

    gb = _safe_grad_norm(backbone_loss, opt_var, eps=eps)
    gg = _safe_grad_norm(gnn_loss, opt_var, eps=eps)

    if gg.item() <= eps * 10:
        return 0.0, {
            "grad_backbone": float(gb.item()),
            "grad_gnn": float(gg.item()),
            "lambda_adapt": 0.0,
            "grad_ratio": 0.0,
        }

    lam = target_ratio * gb / gg
    lam = torch.clamp(lam, min=0.0, max=float(max_lambda))

    grad_ratio = float((lam * gg / gb.clamp_min(eps)).detach().cpu())

    return float(lam.detach().cpu()), {
        "grad_backbone": float(gb.item()),
        "grad_gnn": float(gg.item()),
        "lambda_adapt": float(lam.item()),
        "grad_ratio": grad_ratio,
    }


def safe_official_eval_snapshot(
    P,
    benchmark,
    plc,
    gap=1e-3,
    legalize_copy=True,
    name="snapshot",
    step=None,
    verbose=False,
):
    """
    Evaluates a detached snapshot using official proxy.

    Important:
      - Does NOT modify the optimizer tensor.
      - Optionally legalizes only the copied snapshot.
      - Returns None if invalid or proxy fails.
    """
    try:
        P_eval = P.detach().cpu().clone()

        if legalize_copy:
            P_eval = legalize_hard_macros(P_eval, benchmark, gap=gap)

        valid, violations = validate_placement(P_eval, benchmark)

        if not valid:
            if verbose:
                print(f"    checkpoint {name} step={step}: invalid {violations}")
            return None

        costs = compute_proxy_cost(P_eval, benchmark, plc)

        if not is_valid_proxy_cost(costs):
            if verbose:
                print(f"    checkpoint {name} step={step}: bad cost {costs}")
            return None

        proxy = float(costs["proxy_cost"])

        out = {
            "name": name,
            "step": step,
            "proxy": proxy,
            "costs": costs,
            "pos": P_eval.detach().cpu().clone(),
            "valid": True,
            "violations": violations,
        }

        if verbose:
            print(
                f"    checkpoint {name} step={step}: "
                f"proxy={proxy:.6f} "
                f"wl={float(costs.get('wirelength_cost', float('nan'))):.6f} "
                f"dens={float(costs.get('density_cost', float('nan'))):.6f} "
                f"cong={float(costs.get('congestion_cost', float('nan'))):.6f}"
            )

        return out

    except Exception as exc:
        if verbose:
            print(f"    checkpoint {name} step={step}: failed {type(exc).__name__}: {exc}")
        return None


def maybe_update_official_best(best_ckpt, P, benchmark, plc, gap, name, step, legalize_copy=True, verbose=False):
    ckpt = safe_official_eval_snapshot(
        P=P,
        benchmark=benchmark,
        plc=plc,
        gap=gap,
        legalize_copy=legalize_copy,
        name=name,
        step=step,
        verbose=verbose,
    )

    if ckpt is None:
        return best_ckpt

    if best_ckpt is None or ckpt["proxy"] < best_ckpt["proxy"]:
        if verbose:
            old = float("inf") if best_ckpt is None else best_ckpt["proxy"]
            print(f"    new best official checkpoint: {old:.6f} -> {ckpt['proxy']:.6f}")
        return ckpt

    return best_ckpt

def get_optimizer_lr(opt):
    return float(opt.param_groups[0]["lr"])


def set_optimizer_lr(opt, new_lr):
    for group in opt.param_groups:
        group["lr"] = float(new_lr)


def maybe_decay_lr(opt, bad_count, patience, decay, min_lr, prefix=""):
    cur_lr = get_optimizer_lr(opt)

    if bad_count < patience:
        return bad_count

    if cur_lr <= min_lr + 1e-12:
        return 0

    new_lr = max(min_lr, cur_lr * decay)
    set_optimizer_lr(opt, new_lr)

    print(f"{prefix} adaptive LR decay: {cur_lr:.4e} -> {new_lr:.4e}")

    return 0

# newer version: Hybrid GNN placer.
# newer version: Adam-only, full pairwise overlap penalty, gated GNN congestion residual.
# newer version: Hard legalizer remains final-only.
class HybridAnalyticalPlacer:
    def __init__(
        self,
        trained_model,
        steps=160,
        lr=0.02,
        adam_steps=None,
        lbfgs_steps=0,
        lbfgs_lr=0.0,
        lambda_wl=1.0,
        lambda_dens_start=0.10,
        lambda_dens_end=3.0,
        lambda_cong_start=0.0,
        # newer version: Can be meaningful now because overlap-gating controls when it activates.
        lambda_cong_end=0.05,
        lambda_bound=10.0,
        lambda_repel=1000.0,
        gap=1e-3,
        k_graph=8,
        eval_every=10,
        seeds=(0,),
        use_softplus_cong=False,
        gnn_score_scale=10.0,
        gnn_grad_match=True,
        gnn_grad_target_ratio=0.20,
        gnn_start_frac=0.45,
        gnn_lambda_cap=50.0,
        official_eval_every=10,
        checkpoint_legalize=True,
        checkpoint_verbose=False,
        init_jitter=0.0,
        adaptive_lr=True,
        lr_patience=3,
        lr_decay=0.6,
        min_lr=0.003,
        lr_min_improve=1e-6,
    ):
        self.model = trained_model
        self.steps = steps
        self.lr = lr
        self.adam_steps = steps if adam_steps is None else adam_steps

        self.lbfgs_steps = lbfgs_steps
        self.lbfgs_lr = lbfgs_lr

        self.lw = lambda_wl
        self.lds = lambda_dens_start
        self.lde = lambda_dens_end
        self.lcs = lambda_cong_start
        self.lce = lambda_cong_end
        self.lb = lambda_bound
        self.lr2 = lambda_repel
        self.gap = gap
        self.k = k_graph
        self.eval_every = eval_every
        self.seeds = tuple(seeds)
        self.use_softplus_cong = use_softplus_cong

        self.gnn_score_scale = float(gnn_score_scale)
        self.gnn_grad_match = bool(gnn_grad_match)
        self.gnn_grad_target_ratio = float(gnn_grad_target_ratio)
        self.gnn_start_frac = float(gnn_start_frac)
        self.gnn_lambda_cap = float(gnn_lambda_cap)

        self.official_eval_every = official_eval_every
        self.checkpoint_legalize = bool(checkpoint_legalize)
        self.checkpoint_verbose = bool(checkpoint_verbose)
        self.init_jitter = float(init_jitter)

        self.adaptive_lr = bool(adaptive_lr)
        self.lr_patience = int(lr_patience)
        self.lr_decay = float(lr_decay)
        self.min_lr = float(min_lr)
        self.lr_min_improve = float(lr_min_improve)

    def _hybrid_terms_and_loss(self, P, benchmark, plc, scales, progress, base_lam_d, base_lam_c):
          W, H = float(benchmark.canvas_width), float(benchmark.canvas_height)

          J_wl_raw = smooth_hpwl(P, benchmark, plc) / (W + H)
          J_dens_raw = density_penalty(P, benchmark)

          # Learned congestion residual only.
          J_cong_raw = gnn_congestion_term(
              P,
              benchmark,
              plc,
              self.model,
              k_graph=self.k,
              use_softplus=self.use_softplus_cong,
          )

          J_b = boundary_penalty(P, benchmark, gap=self.gap)
          J_r = full_hard_repulsion_penalty(P, benchmark, gap=self.gap)

          J_wl = J_wl_raw / scales["wl"]
          J_dens = J_dens_raw / scales["dens"]
          J_cong = J_cong_raw / scales["cong"]

          lam_d, lam_c_gate, lam_r = adaptive_hybrid_weights(
              progress=progress,
              J_repel=J_r,
              base_lam_d=base_lam_d,
              base_lam_c=base_lam_c,
              base_lam_r=self.lr2,
          )

          # Backbone stays responsible for placement validity and normal analytical quality.
          backbone_loss = (
              self.lw * J_wl
              + lam_d * J_dens
              + self.lb * J_b
              + lam_r * J_r
          )

          # Scale the learned residual so its gradient is not drowned out.
          gnn_loss = self.gnn_score_scale * J_cong

          late_ramp = smooth_late_ramp_progress(
              progress,
              start_frac=self.gnn_start_frac,
          )

          # Convert the existing overlap gate into a clean multiplier.
          if base_lam_c > 1e-12:
              overlap_gate = lam_c_gate / base_lam_c
          else:
              overlap_gate = 0.0

          gstats = {
              "grad_backbone": 0.0,
              "grad_gnn": 0.0,
              "lambda_adapt": 0.0,
              "grad_ratio": 0.0,
          }

          if self.gnn_grad_match and torch.is_grad_enabled():
              # Cap by base_lam_c so the old lambda_cong schedule still controls max strength.
              lam_c_adapt, gstats = adaptive_gnn_lambda_from_grad(
                  backbone_loss=backbone_loss,
                  gnn_loss=gnn_loss,
                  opt_var=P,
                  target_ratio=self.gnn_grad_target_ratio,
                  max_lambda=self.gnn_lambda_cap,
              )
              lam_c = overlap_gate * late_ramp * lam_c_adapt
          else:
              lam_c = overlap_gate * late_ramp * lam_c_gate

          loss = backbone_loss + lam_c * gnn_loss

          return loss, {
              "J_wl_raw": J_wl_raw,
              "J_dens_raw": J_dens_raw,
              "J_cong_raw": J_cong_raw,
              "J_b": J_b,
              "J_r": J_r,
              "lam_d": lam_d,
              "lam_c": lam_c,
              "lam_c_gate": lam_c_gate,
              "lam_r": lam_r,
              "gnn_score_scale": self.gnn_score_scale,
              "late_ramp": late_ramp,
              "overlap_gate": overlap_gate,
              **gstats,
          }

    def _place_one_seed(self, benchmark, plc, seed=0):
        set_seed(seed)

        P = benchmark.macro_positions.clone().detach()
        P = clamp_centers_to_canvas(
            P,
            benchmark.macro_sizes,
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            self.gap,
        )

        P = P.to(device).requires_grad_(True)

        if getattr(self, "init_jitter", 0.0) > 0:
          gen = torch.Generator(device=P.device)
          gen.manual_seed(int(seed))

          W = float(benchmark.canvas_width)
          H = float(benchmark.canvas_height)

          scale = self.init_jitter * torch.tensor([W, H], device=P.device)
          noise = torch.randn(P.shape, generator=gen, device=P.device) * scale

          with torch.no_grad():
              P.add_(noise)
              P[:] = clamp_centers_to_canvas(
                  P,
                  benchmark.macro_sizes.to(P.device),
                  W,
                  H,
                  self.gap,
              )

        movable = benchmark.get_movable_mask().to(device)
        hard_movable = movable & benchmark.get_hard_macro_mask().to(device)
        original = benchmark.macro_positions.to(device)

        opt = torch.optim.Adam([P], lr=self.lr)
        lr_bad_count = 0


        scales = _objective_scales(P.detach(), benchmark, plc, self.model, self.k)

        best_P = P.detach().clone()
        best_proxy = float("inf")
        best_official = None

        # newer version: Adam-only hybrid optimization.
        # newer version: No hard legalizer inside this loop.
        for t in range(self.adam_steps):
            opt.zero_grad()

            progress = t / max(self.adam_steps - 1, 1)

            base_lam_d = (
                self.lds * (self.lde / self.lds) ** progress
                if self.lds > 0
                else self.lde * progress
            )

            # newer version: Stronger congestion is allowed late, but adaptive_hybrid_weights gates it by overlap.
            base_lam_c = self.lcs + (self.lce - self.lcs) * progress

            loss, info = self._hybrid_terms_and_loss(
                P=P,
                benchmark=benchmark,
                plc=plc,
                scales=scales,
                progress=progress,
                base_lam_d=base_lam_d,
                base_lam_c=base_lam_c,
            )

            loss.backward()

            with torch.no_grad():
                # newer version: Only movable hard macros are optimized.
                P.grad[~hard_movable] = 0.0

            opt.step()

            with torch.no_grad():
                P[:] = clamp_centers_to_canvas(
                    P,
                    benchmark.macro_sizes.to(P.device),
                    float(benchmark.canvas_width),
                    float(benchmark.canvas_height),
                    self.gap,
                )

                # newer version: Fixed or immovable macros remain exactly fixed.
                P[~movable] = original[~movable]

            with torch.no_grad():
                step_score = legality_first_score(loss, info["J_b"], info["J_r"])
                if math.isfinite(step_score) and step_score < best_proxy:
                    best_proxy = step_score
                    best_P = P.detach().clone()

            if (
                self.official_eval_every is not None
                and self.official_eval_every > 0
                and (t % self.official_eval_every == 0 or t == self.adam_steps - 1)
            ):
                old_best_proxy = float("inf") if best_official is None else best_official["proxy"]
                best_official = maybe_update_official_best(
                    best_ckpt=best_official,
                    P=P,
                    benchmark=benchmark,
                    plc=plc,
                    gap=self.gap,
                    name=f"hybrid_seed{seed}",
                    step=t,
                    legalize_copy=self.checkpoint_legalize,
                    verbose=self.checkpoint_verbose,
                )

                new_best_proxy = float("inf") if best_official is None else best_official["proxy"]

                if self.adaptive_lr:
                    if new_best_proxy < old_best_proxy - self.lr_min_improve:
                        lr_bad_count = 0
                    else:
                        lr_bad_count += 1

                    lr_bad_count = maybe_decay_lr(
                        opt=opt,
                        bad_count=lr_bad_count,
                        patience=self.lr_patience,
                        decay=self.lr_decay,
                        min_lr=self.min_lr,
                        prefix=f"  seed={seed}",
                    )

            if t % 20 == 0 or t == self.adam_steps - 1:
                print(
                    f"  seed={seed} hybrid adam {t:3d} | "
                    f"lr={get_optimizer_lr(opt):.2e} "
                    f"lam_d={info['lam_d']:.4f} "
                    f"lam_c={info['lam_c']:.4f} "
                    f"gate={info['overlap_gate']:.2f} "
                    f"ramp={info['late_ramp']:.2f} "
                    f"gratio={info['grad_ratio']:.5f} "
                    f"lam_r={info['lam_r']:.1f} "
                    f"wl={info['J_wl_raw'].item():.4f} "
                    f"dens={info['J_dens_raw'].item():.4f} "
                    f"cong={info['J_cong_raw'].item():.4f} "
                    f"repel={info['J_r'].item():.8f}"
                )



                  # newer version: Optional LBFGS refinement for hybrid candidate.
        # newer version: This is only for hybrid, not the analytical safety fallback.
        # newer version: PyTorch LBFGS internally calls closure multiple times.
        if self.lbfgs_steps > 0:
            P = best_P.detach().clone().to(device).requires_grad_(True)

            opt_lbfgs = torch.optim.LBFGS(
                [P],
                lr=self.lbfgs_lr,
                max_iter=self.lbfgs_steps,
                history_size=20,
                line_search_fn="strong_wolfe",
            )

            lbfgs_calls = {"count": 0}

            def closure():
                opt_lbfgs.zero_grad()

                loss, info = self._hybrid_terms_and_loss(
                    P=P,
                    benchmark=benchmark,
                    plc=plc,
                    scales=scales,
                    progress=1.0,
                    base_lam_d=self.lde,
                    # newer version: allow GNN in LBFGS, but adaptive_hybrid_weights still gates it by overlap.
                    base_lam_c=self.lce,
                )

                loss.backward()

                with torch.no_grad():
                    P.grad[~hard_movable] = 0.0

                lbfgs_calls["count"] += 1

                return loss

            try:
                opt_lbfgs.step(closure)
            except RuntimeError as exc:
                print(f"  newer version: hybrid LBFGS skipped/failed for seed={seed}: {exc}")

            with torch.no_grad():
                P[:] = clamp_centers_to_canvas(
                    P,
                    benchmark.macro_sizes.to(P.device),
                    float(benchmark.canvas_width),
                    float(benchmark.canvas_height),
                    self.gap,
                )
                P[~movable] = original[~movable]

            # newer version: Evaluate final LBFGS result once without backward.
            with torch.no_grad():
                eval_loss, eval_info = self._hybrid_terms_and_loss(
                    P=P,
                    benchmark=benchmark,
                    plc=plc,
                    scales=scales,
                    progress=1.0,
                    base_lam_d=self.lde,
                    base_lam_c=self.lce,
                )

                final_score = legality_first_score(
                    eval_loss,
                    eval_info["J_b"],
                    eval_info["J_r"],
                )

                print(
                    f"  seed={seed} hybrid LBFGS done | "
                    f"closure_calls={lbfgs_calls['count']} "
                    f"lam_d={eval_info['lam_d']:.4f} "
                    f"lam_c={info['lam_c']:.4f} "
                    f"gate={info['overlap_gate']:.2f} "
                    f"ramp={info['late_ramp']:.2f} "
                    f"gratio={info['grad_ratio']:.3f} "
                    f"lam_r={info['lam_r']:.1f} "
                    f"wl={eval_info['J_wl_raw'].item():.4f} "
                    f"dens={eval_info['J_dens_raw'].item():.4f} "
                    f"cong={eval_info['J_cong_raw'].item():.4f} "
                    f"repel={eval_info['J_r'].item():.8f}"
                )

                if math.isfinite(final_score) and final_score < best_proxy:
                    best_proxy = final_score
                    best_P = P.detach().clone()

        # Prefer best valid official checkpoint over surrogate checkpoint.
        if best_official is not None:
            print(
                f"  seed={seed} hybrid selected official checkpoint "
                f"step={best_official['step']} proxy={best_official['proxy']:.6f}"
            )
            return best_official["pos"], best_official["proxy"]

        return best_P.detach().cpu(), best_proxy

    def place(self, benchmark, plc):
          self.model.to(device).eval()

          best_P = None
          best_proxy = float("inf")

          for seed in self.seeds:
              cand, proxy = self._place_one_seed(benchmark, plc, seed=seed)

              if proxy < best_proxy or best_P is None:
                  best_P = cand
                  best_proxy = proxy

          # Diagnostic: placement before final hard legalization
          P_before_legal = best_P.detach().clone()

          # Final-only hard legalization
          P_final = legalize_hard_macros(best_P, benchmark, gap=self.gap)

          # Diagnostic: how much the legalizer changed the placement
          with torch.no_grad():
              delta_legal = P_final.detach().cpu() - P_before_legal.detach().cpu()
              print(
                  "legalizer movement | "
                  f"L2={delta_legal.norm().item():.6e} "
                  f"max={delta_legal.abs().max().item():.6e} "
                  f"mean={delta_legal.abs().mean().item():.6e}"
              )

          valid, violations = validate_placement(P_final, benchmark)
          print("newer version: hybrid candidate valid:", valid, "violations:", violations)

          try:
              cost = compute_proxy_cost(P_final, benchmark, plc)
              print("newer version: hybrid candidate cost:", cost)
          except Exception as exc:
              print(f"newer version: hybrid proxy cost error: {exc}")

          return P_final


# newer version: Analytical baseline/fallback.
# newer version: Adam-only, full O(N^2) overlap penalty, no LBFGS.
class AnalyticalPlacer:
    def __init__(
        self,
        steps=120,
        lr=0.03,
        adam_steps=None,
        lbfgs_steps=0,          # newer version: accepted for compatibility, ignored.
        lbfgs_lr=0.0,           # newer version: accepted for compatibility, ignored.
        lambda_wl=1.0,
        lambda_dens_start=0.10,
        lambda_dens_end=2.0,
        lambda_bound=10.0,
        lambda_repel=1000.0,
        gap=1e-3,
        eval_every=10,
        seeds=(0,),
        official_eval_every=10,
        checkpoint_legalize=True,
        checkpoint_verbose=False,
        init_jitter=0.0,
        adaptive_lr=True,
        lr_patience=3,
        lr_decay=0.6,
        min_lr=0.003,
        lr_min_improve=1e-6,
    ):
        self.steps = steps
        self.lr = lr
        self.adam_steps = steps if adam_steps is None else adam_steps

        # newer version: LBFGS intentionally disabled for analytical fallback.
        self.lbfgs_steps = 0
        self.lbfgs_lr = 0.0

        self.lw = lambda_wl
        self.lds = lambda_dens_start
        self.lde = lambda_dens_end
        self.lb = lambda_bound
        self.lr2 = lambda_repel
        self.gap = gap
        self.eval_every = eval_every
        self.seeds = tuple(seeds)

        self.official_eval_every = official_eval_every
        self.checkpoint_legalize = bool(checkpoint_legalize)
        self.checkpoint_verbose = bool(checkpoint_verbose)

        self.init_jitter = float(init_jitter)
        self.adaptive_lr = bool(adaptive_lr)
        self.lr_patience = int(lr_patience)
        self.lr_decay = float(lr_decay)
        self.min_lr = float(min_lr)
        self.lr_min_improve = float(lr_min_improve)

    def _analytical_terms_and_loss(self, P, benchmark, plc, scales, progress, base_lam_d):
        W, H = float(benchmark.canvas_width), float(benchmark.canvas_height)

        J_wl_raw = smooth_hpwl(P, benchmark, plc) / (W + H)
        J_dens_raw = density_penalty(P, benchmark)
        J_b = boundary_penalty(P, benchmark, gap=self.gap)

        # newer version: Full exact pairwise overlap penalty for analytical fallback.
        J_r = full_hard_repulsion_penalty(P, benchmark, gap=self.gap)

        J_wl = J_wl_raw / scales["wl"]
        J_dens = J_dens_raw / scales["dens"]

        lam_d, lam_r = adaptive_analytical_weights(
            progress=progress,
            J_repel=J_r,
            base_lam_d=base_lam_d,
            base_lam_r=self.lr2,
        )

        loss = (
            self.lw * J_wl
            + lam_d * J_dens
            + self.lb * J_b
            + lam_r * J_r
        )

        return loss, {
            "J_wl_raw": J_wl_raw,
            "J_dens_raw": J_dens_raw,
            "J_b": J_b,
            "J_r": J_r,
            "lam_d": lam_d,
            "lam_r": lam_r,
        }

    def _place_one_seed(self, benchmark, plc, seed=0):
        set_seed(seed)

        P = benchmark.macro_positions.clone().detach()
        P = clamp_centers_to_canvas(
            P,
            benchmark.macro_sizes,
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            self.gap,
        )

        P = P.to(device).requires_grad_(True)

        if getattr(self, "init_jitter", 0.0) > 0:
          gen = torch.Generator(device=P.device)
          gen.manual_seed(int(seed))

          W = float(benchmark.canvas_width)
          H = float(benchmark.canvas_height)

          scale = self.init_jitter * torch.tensor([W, H], device=P.device)
          noise = torch.randn(P.shape, generator=gen, device=P.device) * scale

          with torch.no_grad():
              P.add_(noise)
              P[:] = clamp_centers_to_canvas(
                  P,
                  benchmark.macro_sizes.to(P.device),
                  W,
                  H,
                  self.gap,
              )

        movable = benchmark.get_movable_mask().to(device)
        hard_movable = movable & benchmark.get_hard_macro_mask().to(device)
        original = benchmark.macro_positions.to(device)

        opt = torch.optim.Adam([P], lr=self.lr)
        lr_bad_count = 0
        scales = _objective_scales(P.detach(), benchmark, plc, trained_model=None)

        best_P = P.detach().clone()
        best_proxy = float("inf")
        best_official = None

        # newer version: Adam-only optimization.
        # newer version: Full O(N^2) overlap penalty runs here for analytical validity.
        for t in range(self.adam_steps):
            opt.zero_grad()

            progress = t / max(self.adam_steps - 1, 1)
            base_lam_d = (
                self.lds * (self.lde / self.lds) ** progress
                if self.lds > 0
                else self.lde * progress
            )

            loss, info = self._analytical_terms_and_loss(
                P=P,
                benchmark=benchmark,
                plc=plc,
                scales=scales,
                progress=progress,
                base_lam_d=base_lam_d,
            )

            loss.backward()

            with torch.no_grad():
                P.grad[~hard_movable] = 0.0

            opt.step()

            with torch.no_grad():
                P[:] = clamp_centers_to_canvas(
                    P,
                    benchmark.macro_sizes.to(P.device),
                    float(benchmark.canvas_width),
                    float(benchmark.canvas_height),
                    self.gap,
                )

                # newer version: fixed/immovable macros stay exactly fixed.
                P[~movable] = original[~movable]

            with torch.no_grad():
                step_score = legality_first_score(loss, info["J_b"], info["J_r"])
                if math.isfinite(step_score) and step_score < best_proxy:
                    best_proxy = step_score
                    best_P = P.detach().clone()

            if (
                self.official_eval_every is not None
                and self.official_eval_every > 0
                and (t % self.official_eval_every == 0 or t == self.adam_steps - 1)
            ):
                old_best_proxy = float("inf") if best_official is None else best_official["proxy"]

                best_official = maybe_update_official_best(
                    best_ckpt=best_official,
                    P=P,
                    benchmark=benchmark,
                    plc=plc,
                    gap=self.gap,
                    name=f"analytical_seed{seed}",
                    step=t,
                    legalize_copy=self.checkpoint_legalize,
                    verbose=self.checkpoint_verbose,
                )

                new_best_proxy = float("inf") if best_official is None else best_official["proxy"]

                if self.adaptive_lr:
                    if new_best_proxy < old_best_proxy - self.lr_min_improve:
                        lr_bad_count = 0
                    else:
                        lr_bad_count += 1

                    lr_bad_count = maybe_decay_lr(
                        opt=opt,
                        bad_count=lr_bad_count,
                        patience=self.lr_patience,
                        decay=self.lr_decay,
                        min_lr=self.min_lr,
                        prefix=f"  seed={seed}",
                    )

            if t % 20 == 0 or t == self.adam_steps - 1:
                print(
                    f"  seed={seed} adam {t:3d} | "
                    f"lr={get_optimizer_lr(opt):.2e} "
                    f"lam_d={info['lam_d']:.4f} lam_r={info['lam_r']:.1f} "
                    f"wl={info['J_wl_raw'].item():.4f} "
                    f"dens={info['J_dens_raw'].item():.4f} "
                    f"repel={info['J_r'].item():.8f}"
                )

        if best_official is not None:
            print(
                f"  seed={seed} analytical selected official checkpoint "
                f"step={best_official['step']} proxy={best_official['proxy']:.6f}"
            )
            return best_official["pos"], best_official["proxy"]

        return best_P.detach().cpu(), best_proxy

    def _run_attempt(self, benchmark, plc, attempt_name, seeds):
        print(
            f"\nnewer version: analytical attempt={attempt_name} | "
            f"lr={self.lr} steps={self.adam_steps} "
            f"dens_end={self.lde} repel={self.lr2} gap={self.gap} seeds={seeds}"
        )

        best_P = None
        best_proxy = float("inf")

        for seed in seeds:
            cand, proxy = self._place_one_seed(benchmark, plc, seed=seed)
            if proxy < best_proxy or best_P is None:
                best_P = cand
                best_proxy = proxy

        # newer version: Final-only hard legalization.
        P_final = legalize_hard_macros(best_P, benchmark, gap=self.gap)

        valid, violations = validate_placement(P_final, benchmark)
        print(f"newer version: analytical attempt={attempt_name} valid={valid} violations={violations}")

        if valid:
            try:
                cost = compute_proxy_cost(P_final, benchmark, plc)
                print(f"newer version: analytical attempt={attempt_name} cost={cost}")
            except Exception as exc:
                print(f"newer version: analytical attempt={attempt_name} valid but proxy failed: {exc}")

            return P_final, True, violations

        return P_final, False, violations

    def place(self, benchmark, plc):
        # newer version: Save original settings so retry ladder can mutate safely.
        original_lr = self.lr
        original_steps = self.steps
        original_adam_steps = self.adam_steps
        original_lde = self.lde
        original_lr2 = self.lr2
        original_gap = self.gap
        original_seeds = self.seeds

        failures = []

        # newer version: Finite retry ladder.
        # newer version: Return immediately after first valid placement.
        retry_configs = [
            {
                "name": "normal_full_pair_adam",
                "lr": original_lr,
                "adam_steps": original_adam_steps,
                "dens_end": original_lde,
                "repel": original_lr2,
                "gap": original_gap,
                "seeds": original_seeds,
            },
            {
                "name": "stronger_repel",
                "lr": min(original_lr, 0.02),
                "adam_steps": max(original_adam_steps, 180),
                "dens_end": max(original_lde, 3.0),
                "repel": max(original_lr2 * 3.0, 3000.0),
                "gap": original_gap,
                "seeds": tuple(sorted(set(original_seeds + (0, 1)))),
            },
            {
                "name": "spread_more",
                "lr": min(original_lr, 0.02),
                "adam_steps": max(original_adam_steps, 220),
                "dens_end": max(original_lde * 2.0, 6.0),
                "repel": max(original_lr2 * 5.0, 5000.0),
                "gap": original_gap,
                "seeds": tuple(sorted(set(original_seeds + (0, 1, 2)))),
            },
            {
                "name": "safe_spacing",
                "lr": min(original_lr, 0.015),
                "adam_steps": max(original_adam_steps, 250),
                "dens_end": max(original_lde * 3.0, 10.0),
                "repel": max(original_lr2 * 8.0, 8000.0),
                "gap": original_gap * 2.0,
                "seeds": tuple(sorted(set(original_seeds + (0, 1, 2, 3)))),
            },
            {
                "name": "last_resort_validity",
                "lr": min(original_lr, 0.01),
                "adam_steps": max(original_adam_steps, 300),
                "dens_end": max(original_lde * 5.0, 15.0),
                "repel": max(original_lr2 * 12.0, 12000.0),
                "gap": original_gap * 3.0,
                "seeds": tuple(sorted(set(original_seeds + (0, 1, 2, 3, 4)))),
            },
        ]

        try:
            for cfg in retry_configs:
                self.lr = cfg["lr"]
                self.steps = cfg["adam_steps"]
                self.adam_steps = cfg["adam_steps"]
                self.lde = cfg["dens_end"]
                self.lr2 = cfg["repel"]
                self.gap = cfg["gap"]

                P_final, valid, violations = self._run_attempt(
                    benchmark=benchmark,
                    plc=plc,
                    attempt_name=cfg["name"],
                    seeds=cfg["seeds"],
                )

                if valid:
                    print(f"newer version: selected analytical fallback attempt={cfg['name']}")
                    return P_final

                failures.append((cfg["name"], violations))

            raise RuntimeError(
                "newer version: analytical Adam retry ladder failed to find a valid placement. "
                f"failures={failures}"
            )

        finally:
            # newer version: Restore original object settings.
            self.lr = original_lr
            self.steps = original_steps
            self.adam_steps = original_adam_steps
            self.lde = original_lde
            self.lr2 = original_lr2
            self.gap = original_gap
            self.seeds = original_seeds

class PICongesionGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        # def _head():
        #     return nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        #         nn.Linear(hidden_dim, 1))
        # self.head_cong = _head()
        # self.head_wl   = _head()
        # self.head_dens = _head()
        self.head_cong = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, data):
        x,ei,batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x,ei))
        x = F.relu(self.conv2(x,ei))
        x = F.relu(self.conv3(x,ei))
        g = global_mean_pool(x, batch)
        return self.head_cong(g).squeeze(-1)
        # return (self.head_cong(g).squeeze(-1),
        #         self.head_wl(g).squeeze(-1),
        #         self.head_dens(g).squeeze(-1))
