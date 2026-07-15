"""Trainer for the 2,000 step speedrun.

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

Every deviation from the baseline is a flag with a baseline-preserving default,
so any run in RUNLOG.md can be reproduced exactly:

    # exact baseline
    python train.py --data llm_handout/data/train_corpus.txt --steps 2000 \
        --out ckpt_baseline.pt
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device, generator):
    ix = torch.randint(len(ids) - block - 1, (batch,), generator=generator)
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, args):
    """Warmup then decay. step is 1-based."""
    if args.warmup > 0 and step <= args.warmup:
        return args.lr * step / args.warmup
    if args.schedule == "none":
        return args.lr
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    progress = min(1.0, max(0.0, progress))
    if args.schedule == "cosine":
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif args.schedule == "linear":
        scale = 1.0 - progress
    else:
        raise ValueError(args.schedule)
    return args.min_lr + (args.lr - args.min_lr) * scale


def build_optimizer(model, args):
    """Returns a list of optimizers; all of them get stepped each step."""
    if args.opt == "adam":
        return [torch.optim.Adam(model.parameters(), lr=args.lr)]

    if args.opt == "muon":
        from optim import Muon, split_params
        muon_p, other_p = split_params(model)
        n_m = sum(p.numel() for p in muon_p)
        n_o = sum(p.numel() for p in other_p)
        print(f"muon: {len(muon_p)} matrices / {n_m:,} params | "
              f"adamw: {len(other_p)} tensors / {n_o:,} params")
        return [Muon(muon_p, lr=args.muon_lr, momentum=args.muon_momentum,
                     ns_steps=args.ns_steps, weight_decay=args.wd),
                torch.optim.AdamW(other_p, lr=args.lr, weight_decay=0.0,
                                  betas=(args.beta1, args.beta2))]

    # Decay only matrices; leave norms, biases and embeddings undecayed.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 and "emb" not in name else no_decay).append(p)
    return [torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=None,
                    help="sequence length; default = Config.block_size")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=0.0)
    ap.add_argument("--opt", choices=["adam", "adamw", "muon"], default="adam")
    ap.add_argument("--muon_lr", type=float, default=0.02,
                    help="Muon lr for 2D hidden matrices (--lr still drives "
                         "the AdamW group holding embeddings/norms/biases)")
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--ns_steps", type=int, default=5)
    ap.add_argument("--schedule", choices=["none", "cosine", "linear"],
                    default="none")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.0, help="0 = off")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--init_std", type=float, default=0.05)
    ap.add_argument("--resid_scale", action="store_true",
                    help="scale residual projections by 1/sqrt(2*n_layer)")
    ap.add_argument("--tie", action="store_true", help="tie head to tok_emb")
    ap.add_argument("--norm", choices=["layer", "rms"], default="layer")
    ap.add_argument("--pos", choices=["learned", "rope"], default="learned")
    ap.add_argument("--mlp", choices=["gelu", "swiglu"], default="gelu")
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_bytes = len(text.encode("utf-8"))
    print(f"corpus: {n_bytes:,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size}, {n_bytes/len(ids):.2f} bytes/token)")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    if args.block is not None:
        cfg.block_size = args.block
    cfg.tie_weights = args.tie
    cfg.init_std = args.init_std
    cfg.resid_scale = args.resid_scale
    cfg.norm, cfg.pos, cfg.mlp = args.norm, args.pos, args.mlp
    if args.n_layer is not None:
        cfg.n_layer = args.n_layer
    if args.n_embd is not None:
        cfg.n_embd = args.n_embd
    if args.n_head is not None:
        cfg.n_head = args.n_head
    model = GPT(cfg).to(device)
    n = model.n_params()
    seen = args.steps * args.batch * cfg.block_size
    print(f"model: {n:,} params ({n/MAX_PARAMS*100:.1f}% of cap) | "
          f"block {cfg.block_size} batch {args.batch} | "
          f"will see {seen:,} tokens = {seen/len(ids)*100:.1f}% of corpus")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params (got {n:,})"

    opts = build_optimizer(model, args)
    # Each group keeps its own base LR (Muon's differs from AdamW's by ~20x),
    # and the schedule scales all of them by the same factor.
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        scale = lr_at(step, args) / args.lr
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale
        x, y = get_batch(ids, cfg.block_size, args.batch, device, gen)
        _, loss = model(x, y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for o in opts:
            o.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  scale {scale:.3f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses,
                # plain dict, not Namespace: evaluate.py loads with
                # weights_only=True, which rejects arbitrary objects
                "args": {k: v for k, v in vars(args).items()
                         if isinstance(v, (int, float, str, bool))}},
               args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)  "
          f"final-100 loss {sum(losses[-100:])/100:.4f}")


if __name__ == "__main__":
    main()
