# NOTES — best configuration and why it works

*(Brief caps this at 10 sentences. Final numbers are filled from the last run;
see RUNLOG.md for every run, including the failures.)*

1. **Best configuration:** byte-level BPE trained on the corpus (vocab 4,096,
   3.49 bytes/token) with the output head tied to the token embedding, block
   512, RMSNorm + SwiGLU (hidden 8/3·d), 4 layers × 4 heads × n_embd 160,
   AdamW at peak lr 1e-3 with 100-step warmup and cosine decay to zero, weight
   decay 0.1 on matrices only, grad clip 1.0, β₂ 0.95, init N(0, 0.05),
   dropout 0 — 2,000 steps, CPU, under the 2,000,000-parameter cap.
2. The whole result follows from one measurement on the baseline: its loss is
   still falling steeply at step 2,000 and it reads only 28% of its corpus, so
   the binding constraint is the **step cap**, never model capacity — which
   means every step must be made to read more text, and tuning constants cannot
   help.
3. **The tokenizer is the dominant lever (−0.2014 bpb):** the corpus is 14.1%
   Devanagari and the eval text 20.5%, and UTF-8 spends three bytes per
   Devanagari character, so a byte tokenizer burns its context budget precisely
   where the evaluation is densest.
4. BPE compresses 3.49 bytes/token, which converts 28% corpus coverage into
   roughly two full epochs and widens the model's view from 128 bytes to ~1,685
   — the same 2,000 steps simply see far more of the world.
5. **Weight tying is not an optimisation here, it is what makes BPE legal:**
   untied, vocab 4,096 costs 2,589,120 parameters (129% of cap); tied, the same
   model is 1,933,760 — so the starter's `tie_weights = False` is what makes the
   single biggest win look impossible.
6. **An LR schedule matters more here than in a normal run (−0.0983):** with no
   option to train longer, the learning rate must complete its entire job inside
   2,000 steps, so the baseline's defect was never the value 3e-4 but that it
   never changed.
7. Context width paid twice — block 128→256 (−0.0789) and 256→512 (−0.0342) —
   into clear diminishing returns, which is where we stopped.
8. **Every attempt to tune a baseline constant failed** (init 0.02 + residual
   scaling: +0.153; init 0.08: +0.070; lr 2e-3: +0.008), so the model shipped
   here is architecturally the baseline's own — the starter is "mediocre on
   purpose" in exactly three places (schedule, tokenizer, context) and well-set
   everywhere else.
9. RMSNorm + SwiGLU won narrowly (−0.0066) and, deliberately param-matched
   *downward*, did so with 5,728 fewer parameters — so the gain is the gating,
   not extra capacity.
10. **The rule that predicts all of it:** under a hard step cap, techniques
    whose benefit is *amortised over steps* (small init, residual scaling) never
    collect and therefore lose, while techniques available *immediately*
    (compression, an annealed schedule, gating) win — that distinction, not
    model size, is what this speedrun rewards.
