# RUNLOG — 2,000 Step LLM Speedrun

Score = bits per byte (bpb) on held-out text, via the official scorer:
`python evaluate.py --checkpoint ckpt.pt --text_file llm_handout/data/dev_eval.txt`
Lower is better. All runs: 2,000 steps max, 2,000,000 params max, CPU only.

Each entry records who drove the decision (H = Devangan, M = Claude Code) so
SUMMARY.html's machine/human split is recorded as it happens rather than
reconstructed afterwards.

---

## Run 0 — baseline, untouched

**Hypothesis:** none. Establish the reference number before changing anything,
so every later claim has something to be measured against.

**What changed:** nothing. Starter code exactly as shipped.

**Config:** vocab 256 (raw UTF-8 bytes), block 128, batch 8, n_layer 4,
n_head 4, n_embd 160, dropout 0.0, tie_weights False, Adam constant lr 3e-4,
no warmup, no schedule, no weight decay, no grad clipping, init N(0, 0.05)
flat across every Linear and Embedding. Seed 1337.

**Result:** dev **bpb 2.3718** — 1,339,840 params, 2,000 steps.
Train: 106s total, 53 ms/step. Final train loss 1.7315 nats/token.

**Observations (measured, not inferred):**
- 1,339,840 params against a 2,000,000 cap — ~660K unused headroom.
- Corpus tokenizes to 7,318,592 tokens at vocab 256, i.e. 1 token = 1 byte.
  The run sees 2000 x 8 x 128 = 2,048,000 tokens = **~28% of the corpus**.
  The model never reads roughly three-quarters of its training data.
- Loss still falling steeply at the cap: 1.83 (step 1500) -> 1.73 (step 2000).
  Not converged; the step budget ends mid-descent.
- Train loss 1.7315 nats = 2.498 bits/token = 2.498 bpb (byte tokenizer makes
  these identical). Dev bpb 2.3718 is lower because the training figure is a
  running average over a still-improving run, not the final model's rate.
- 53 ms/step at 1024 tokens/step leaves large compute headroom inside the
  2,000-step cap: steps are capped, tokens per step are not.

**Conclusion:** _[H to write — this is graded reasoning and should be yours]_

**Attribution:** M ran the baseline and collected the numbers. No judgement
exercised yet; nothing was changed.

---

## Candidate levers (identified, not yet tested)

Ranked by expected effect. Nothing here is a result — these are hypotheses
awaiting runs, listed so the log shows what was considered and what was
rejected, not only what happened to work.

1. **Tokenizer -> byte-level BPE.** Corpus is 14.1% Devanagari, dev is 20.5%.
   Byte tokenizer spends 3 tokens per Devanagari char, so it burns the most
   context exactly where the eval is densest. BPE raises bytes-per-token, which
   directly buys corpus coverage inside a fixed step budget.
2. **Weight tying.** Costed exactly: at vocab 4096 / n_embd 160 / 4 layers /
   block 256, tied = 1,933,760 params (fits); untied adds 655,360 (over cap).
   `tie_weights = False` is what makes BPE look impossible. It is not optional.
3. **LR schedule + warmup.** Constant 3e-4 with a visibly unconverged curve is
   the most obvious defect in the baseline.
4. **Init.** Flat N(0, 0.05) for every tensor, no residual-depth scaling.
5. **Tokens per step.** batch/block are free within the step cap; compute
   headroom measured above says we can afford ~4-8x.
6. **AdamW + weight decay + grad clipping.** Baseline has none of the three.
7. **Architecture** (RMSNorm / RoPE / SwiGLU). Cheapest last: RoPE also frees
   the pos_emb params.

---
