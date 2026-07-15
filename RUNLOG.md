# RUNLOG — 2,000 Step LLM Speedrun

Score = bits per byte (bpb) on held-out text, via the official scorer:
`python evaluate.py --checkpoint ckpt.pt --text_file llm_handout/data/dev_eval.txt`
Lower is better. All runs: 2,000 steps max, 2,000,000 params max, CPU only.


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

**Conclusion:** The step budget, not the model, is the binding constraint. Two
independent things say so: the loss is still falling at step 2000, and the run
only ever reads 28% of the corpus. That reframes the problem — the question is
not "how do I train this model better" but "how much text and how much context
can I get through 2,000 fixed steps". Every lever below is judged by that.

---

## Control — refactored trainer, baseline flags

**Hypothesis:** the rewrite of train.py (flags, param groups, LR schedule hooks)
changed nothing behavioural. If it did, every later comparison is worthless.

**What changed:** code structure only. Same hyperparameters as Run 0.
One deliberate difference: batch sampling moved to its own `torch.Generator`,
decoupling data order from model-init RNG, so the batch sequence is not
bit-identical to Run 0.

**Result:** dev **bpb 2.3710** vs Run 0's 2.3718. Delta 0.0008.

**Conclusion:** The refactor is faithful; the 0.0008 gap is the batch-order
change, and it sets the noise floor for this setup. Any effect below ~0.001 bpb
is not distinguishable from seed noise and will not be claimed as a result.

---

## Run 1 — fix the optimizer, nothing else

**Hypothesis:** Run 0's curve is still descending at the cap because a constant
3e-4 never anneals. With steps hard-capped, an LR schedule is worth more here
than in a normal run: there is no "train longer" to fall back on, so the LR must
do its whole job inside 2,000 steps. Warmup should permit a higher peak safely.

**What changed:** Adam -> AdamW, lr 3e-4 -> 1e-3, constant -> cosine decay to 0,
100-step warmup, weight decay 0.1 (matrices only, not norms/biases/embeddings),
grad clipping 1.0, beta2 0.999 -> 0.95. Tokenizer, model and data untouched.

**Result:** dev **bpb 2.2727** (from 2.3710). **Delta -0.0983.**
Final-100 train loss 1.6559 vs 1.7609. 63 ms/step.

**Conclusion:** Confirmed, and cheap — ~4% of the baseline's bpb for zero
parameters and zero extra compute. The annealed tail is doing the work: the
model spends its last few hundred steps consolidating at a small LR instead of
bouncing around at 3e-4. This is the "free" part of the budget; everything after
it has to buy its gains with parameters or wall-clock.

---

## Run 2 — add the standard init fix (FAILED, kept deliberately)

**Hypothesis:** Run 0's init is flat N(0, 0.05) on every tensor with no
depth-awareness. The textbook GPT-2 recipe is std 0.02 plus scaling residual
projections by 1/sqrt(2*n_layer) so the residual stream's variance does not grow
with depth. Expected a small but reliable gain on top of Run 1.

**What changed:** init_std 0.05 -> 0.02, plus 1/sqrt(2*n_layer) scaling on
attn.proj and mlp[2]. Everything else identical to Run 1.

**Result:** dev **bpb 2.4255** (from Run 1's 2.2727). **Delta +0.1528 — a large
regression, and worse than the untouched baseline's 2.3718.**
Final-100 train loss 1.8200 vs Run 1's 1.6559 — behind for the whole run, not
just at the end.

**Conclusion:** The textbook recipe is tuned for a regime this run is not in.
Both changes shrink the init, and they compound: std drops 2.5x (0.05 -> 0.02),
then residual projections take a further 1/sqrt(8) = 0.354x, landing at
std 0.0071 — about 7x smaller than baseline. That makes every block start near
the identity, which is exactly what you want when training a deep network for
many thousands of steps, because it keeps the residual stream well-conditioned
long enough to learn. Here there is no "long enough": with 2,000 steps and only
4 layers, there is no depth pathology to protect against, and the model instead
burns a large share of its fixed budget just re-growing weights to a useful
scale before it can fit anything. Small init is a stability investment that pays
back over a long run — and this run is too short to collect. Directionally, that
is the same lesson as Run 1 (the step cap dominates), arrived at from the
opposite side.

**Follow-up required:** this bundled two changes, so it does not yet say *which*
one hurt, or whether one of them helps alone. Runs 2a/2b disentangle them.

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
