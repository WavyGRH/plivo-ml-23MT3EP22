"""Muon: momentum orthogonalized by Newton-Schulz.

Pure PyTorch + stdlib, no external packages, CPU-safe. Implemented from the
published method (Jordan et al., the nanoGPT speedrun optimizer); no code or
weights are downloaded or imported.

The idea: Adam rescales each weight *element* by its own gradient history, so a
matrix whose gradient is dominated by one direction takes a step that is
effectively low-rank — most of the update's energy goes into a single direction
and the other directions barely move. Muon instead takes the momentum matrix and
replaces it with the nearest semi-orthogonal matrix (all singular values driven
to 1) before stepping. Every direction then gets an equally-sized step, so a
single step teaches the layer more.

Why it might suit this problem specifically: the cap here is *optimizer steps*,
not compute or wall-clock. Anything that extracts more progress per step is
worth more than usual, and Muon's whole claim is progress-per-step. It is also
where a run can plausibly beat a well-tuned AdamW rather than tie it.

Applies to 2D hidden matrices only. Embeddings, the tied head, norms and biases
keep AdamW: orthogonalization is meaningless for a 1D tensor, and an embedding
table's rows are looked up independently rather than acting as a linear map.
"""
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximate the orthogonal factor of G (i.e. U @ V.T of its SVD) with a
    quintic Newton-Schulz iteration. Coefficients from the reference method:
    tuned so the iteration converges fast without ever computing an SVD, which
    would dominate CPU time here."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        super().__init__(list(params), dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mom, nesterov = group["momentum"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # Newton-Schulz returns singular values ~1 regardless of the
                # matrix's shape, so rescale to keep the update's RMS
                # comparable across differently-shaped layers.
                g = g * max(1.0, g.size(0) / g.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(g, alpha=-group["lr"])


def split_params(model):
    """(muon_params, adamw_params): 2D hidden matrices vs everything else."""
    muon, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "emb" not in name and "head" not in name:
            muon.append(p)
        else:
            other.append(p)
    return muon, other
