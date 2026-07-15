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

**Conclusion (FIRST ATTEMPT — later falsified, kept to show the correction):**
"The textbook recipe is tuned for a regime this run is not in. Both changes
shrink the init and compound: std drops 2.5x (0.05 -> 0.02), then residual
projections take a further 1/sqrt(8) = 0.354x, landing at std 0.0071 — ~7x
smaller than baseline. Every block starts near the identity, which is what you
want when training deep networks for many thousands of steps. Here there is no
'long enough': the model burns a large share of a fixed budget re-growing
weights to a useful scale before it can fit anything."

**That explanation makes a testable prediction: if the problem is that the init
is too small, then a LARGER init should win.** Run 6 tested exactly that with
init_std 0.08 and got **2.0626 — also worse** than 0.05's 1.9924.

**Revised conclusion:** the prediction failed, so the explanation is wrong, or
at least unsupported. What the three points actually show is a local optimum:

| init_std | dev bpb | vs 0.05 |
|---------:|--------:|--------:|
| 0.02 (+resid_scale) | 2.4255 | +0.4331 |
| **0.05 (baseline)** | **1.9924** | — |
| 0.08 | 2.0626 | +0.0702 |

Moving in *either* direction hurts, asymmetrically — smaller is punished harder
than larger. The honest read is that the baseline's init is not a defect at all;
0.05 is close to well-chosen for this width and depth, and I assumed otherwise
because flat N(0, 0.05) does not match the GPT-2 convention I pattern-matched
to. Importing a recipe from a different scale, without checking that its
premises hold here, is the actual mistake — and it is the same mistake in both
directions, which is why the inversion failed too. The brief calls the baseline
"mediocre on purpose"; that does not mean every line of it is wrong, and
assuming so cost two runs.

**Limitation of this run:** it changed two things at once, so **the -0.1528 is
attributed to the bundle, not to either half** — this entry cannot say whether
init_std 0.02, the residual scaling, or their interaction did the damage.
Bundling was a methodology error; the brief says change one thing at a time, and
the cost is exactly this: a large, reproducible result that cannot be assigned a
cause. Run 6 recovers part of it — init_std 0.08 alone, without residual
scaling, also loses — which establishes that init_std is independently sensitive
in both directions. Residual scaling in isolation is not something this log has
a measurement for, and no claim is made about it.

---

## Run 5 — LR retune for the BPE regime (FAILED)

**Hypothesis:** lr 1e-3 was tuned in Run 1 against the *byte* model. Run 3
changed the vocabulary 16-fold and tied the head to the embedding, which is a
materially different loss surface — the earlier optimum should have moved.

**What changed:** lr 1e-3 -> 2e-3. Everything else as Run 4.

**Result:** dev **bpb 2.0003** vs Run 4's 1.9924. **Delta +0.0079.**

**Conclusion:** Rejected, but only just — +0.0079 is small, though ~10x the
0.0008 noise floor established by the Control, so it is a real if minor
regression. The premise was reasonable and simply did not hold: 1e-3 survives
the tokenizer change intact. Worth noting the asymmetry in what tuning bought
here — the *schedule* (Run 1) was worth -0.0983, while the *peak value* it
anneals from is already right. The baseline's defect was never the number 3e-4;
it was that the number never changed.

---

## Run 6 — invert Run 2: bigger init, not smaller (FAILED)

**Hypothesis:** stated above, from Run 2's first conclusion. If a short budget
punishes small init because the model wastes steps growing weights, then
init_std 0.08 should beat 0.05.

**What changed:** init_std 0.05 -> 0.08, no residual scaling. Otherwise Run 4.

**Result:** dev **bpb 2.0626** vs 1.9924. **Delta +0.0702.**

**Conclusion:** Prediction falsified — see Run 2's revised conclusion, which
this run forced. The value of this run is entirely negative-result: it killed a
plausible, confident, wrong story. Had it not been run, Run 2's explanation
would have gone into the write-up sounding authoritative and being unsupported.

**Broader pattern (runs 2, 5, 6):** every attempt to tune a baseline
*hyperparameter* has failed. All the gains so far come from the schedule, the
tokenizer, and context width — structural choices, not constants. The baseline
is mediocre precisely and only where the brief hints it is.

---

## Run 3 — byte-level BPE (vocab 4096) + weight tying

**Hypothesis:** Run 0 established that the step budget, not the model, is
binding: 2,000 steps x 8 x 128 reads only 28% of the corpus, and the model's
whole view of the world is 128 bytes wide. A byte tokenizer is the cause of
both. It spends 3 tokens per Devanagari character — and dev is 20.5% Devanagari
against train's 14.1%, so the eval is *denser* in exactly the text the
tokenizer handles worst. BPE should raise bytes-per-token ~3x, which buys
corpus coverage and context width simultaneously without touching the model.

Tying is not a separate idea here, it is the enabling constraint: an untied
head at vocab 4096 costs 655,360 params and lands at 2,589,120 = 129% of cap.
Tied, the same config is 1,913,280. **BPE is only legal because of tying** —
which is what makes the starter's `tie_weights = False` (commented "one of many
things worth questioning") an effective trap. Left alone, it makes the single
biggest win look impossible.

**What changed:** tokenizer 256-byte -> BPE vocab 4096 trained on
train_corpus.txt only (67,014 unique pre-token chunks, 3,840 merges, 15s);
tie_weights False -> True. Block stays 128. Optimizer as Run 1.

**Result:** dev **bpb 2.0713** (from Run 1's 2.2727). **Delta -0.2014** — twice
the optimizer's gain. 1,913,280 params (95.7% of cap). 87 ms/step (up from 63:
the vocab-4096 output head is ~16x the byte model's matmul).

Measured compression: 3.485 bytes/token on train, 3.292 on dev. Corpus
7,318,592 tokens -> 2,099,764. Eval text 159,225 -> 48,371 tokens.

**Losslessness:** verified before the run, not after — a lossy tokenizer is a
disqualification, not a bug. test_tokenizer.py round-trips both corpora exactly
plus 18 edge cases, including emoji, CJK and Arabic (scripts absent from the
training mix), lone combining marks, control bytes and whitespace runs. All
exact. The byte fallback is structural: the base vocabulary is all 256 byte
values, so every byte sequence is representable by construction.

**Conclusion:** Confirmed, and it is the dominant lever. Note *why* it works —
not because compression is inherently good, but because the binding constraint
was step count. The same 2,000 steps now read 3.5x more text, and the model's
context covers ~450 bytes instead of 128. Per-token train loss *rose* (4.70 vs
1.66) and that is meaningless: a vocab-4096 token carries ~3.5 bytes, so
per-token loss is not comparable across tokenizers. This is precisely why the
brief scores bits per *byte*. Any comparison of train loss across a tokenizer
change is a category error.

---

## Run 4 — block 128 -> 256

**Hypothesis:** Run 3 bought context via bytes-per-token; block_size buys it
directly. The scorer slides a `cfg.block_size` window with 50% carry-over, so a
longer block improves the *eval* as well as training. Cost is params
(block_size x n_embd for the learned pos_emb table) and compute (attention is
quadratic in T).

**What changed:** block_size 128 -> 256. Nothing else.

**Result:** dev **bpb 1.9924** (from 2.0713). **Delta -0.0789.**
1,933,760 params (96.7% of cap). 193 ms/step (from 87 — a 2.2x cost).

**Conclusion:** Confirmed and worth its cost. Combined with BPE, effective
context is now ~256 x 3.29 = ~840 bytes vs the baseline's 128 — a 6.6x widening,
and the model sees ~1.95 epochs of the corpus rather than 28% of one. Cumulative
2.3718 -> 1.9924 = **-16%**. The remaining question this raises: at 96.7% of cap
the learned pos_emb table is now buying context at 40,960 params per doubling,
which is why Run 9 tests RoPE — where block_size costs *zero* parameters
(measured: 1,892,800 params at block 256, 512, 1024 and 2048 alike).

---

## Run 7 — block 256 -> 512

**Hypothesis:** Run 4 showed context is worth buying. The question is whether it
keeps paying at 512, or whether ~840 bytes is already enough for this text.
Cost is steep and known: 40,960 more params for the pos_emb table (98.7% of
cap, leaving no headroom for anything else) and quadratic attention.

**What changed:** block_size 256 -> 512. Nothing else.

**Result:** dev **bpb 1.9582** (from 1.9924). **Delta -0.0342.**
1,974,720 params (98.7% of cap). 438 ms/step, 876s total.

**Conclusion:** Confirmed, still paying, but clearly into diminishing returns —
128->256 bought -0.0789, 256->512 bought -0.0342, less than half as much for
2.3x the compute. Effective context is now ~512 x 3.29 = ~1,685 bytes vs the
baseline's 128. Extrapolating, 512->1024 would be worth perhaps -0.015 for ~40
minutes of CPU, which the deadline does not justify — so this is where context
buying stops. The real cost is the pos_emb table: at 98.7% of cap, *nothing
else can be added to this model*. That is what makes RoPE (Run 9) worth testing
even if rotary and learned positions score identically — it would refund 81,920
params and make block_size free.

---

## Run 10 — RMSNorm + SwiGLU (param-matched)

**Hypothesis:** received wisdom says gated MLPs beat GELU at equal parameters,
and RMSNorm is a free simplification. **Predicted this would lose, or land in
the noise** — the same "long-run recipe" reasoning that produced Runs 2 and 6,
both of which failed. Stated the prediction before running it precisely so it
could be wrong on the record.

**What changed:** LayerNorm -> RMSNorm (drops mean-subtraction and bias, 9 norm
layers x 160 = 1,440 params saved) and GELU 4x MLP -> SwiGLU. SwiGLU needs three
matrices where GELU needs two, so the expansion factor drops 4x -> 8/3 to keep
parameters equal: 3*d*(8d/3) == 8d^2. At d=160 that is 426.67, rounded DOWN to
424 (a multiple of 8), leaving SwiGLU 5,120 params *below* GELU. Rounding up to
432 — the first implementation — would have handed it +11,136 params (+0.58%)
and made any win unattributable. Erring low means a win is the gating, not the
budget.

**Result:** dev **bpb 1.9858** vs Run 4's 1.9924 at the same block 256.
**Delta -0.0066**, with **1,928,032 params — 5,728 FEWER than the control.**

**Conclusion:** Confirmed, and the prediction was wrong. The gain is small but
real: 8x the 0.0008 noise floor, and won while *giving up* capacity, which is
the strongest form the result could take. Worth being precise about why the
prediction failed, rather than quietly banking the win. Runs 2 and 6 failed
because init scale is a *dynamics* choice — small init only pays back over many
steps, so a 2,000-step budget never collects. Gating is not that kind of choice:
it changes what the layer can represent per step, and costs nothing to exploit.
"Long-run wisdom" was never the right category — the question is whether a
technique's benefit is amortised over steps (init, residual scaling: loses here)
or available immediately (gating: wins here). That distinction, not the length
of the run, is what predicts transfer.

Practical value beyond bpb: at 98.7% of cap, a change that *refunds* params is
worth more than its score suggests.

---

## Run 9 — RoPE instead of learned positional embeddings

**Hypothesis:** two reasons, one budgetary and one about learning dynamics.
Budget: the learned pos_emb table costs block_size x n_embd, so at block 512 it
is 81,920 params and the model sits at 98.7% of cap with no headroom for
anything else; RoPE has *no* position parameters, and is block-size independent
(measured: 1,892,800 params at block 256, 512, 1024 and 2048 alike). Dynamics:
Run 10's rule says techniques whose benefit is *immediate* win under a step cap
while *amortised* ones lose — and a learned pos_emb is amortised almost by
definition, because every row must be trained before it means anything.

**What changed:** pos = learned -> rope. Block stays 256, so the comparison is
against Run 4's 1.9924 directly. Nothing else touched.

**Result:** dev **bpb 1.8184** (from 1.9924). **Delta -0.1740** — the second
largest single result of the run, behind only BPE. 1,892,800 params: 40,960
*fewer* than the control. 215 ms/step (vs 193 — rotation costs ~11%).

**Conclusion:** Confirmed, and larger than expected. The headline comparison is
the one worth keeping: **RoPE at block 256 (1.8184) beats learned positions at
block 512 (1.9582)** — a better score with half the context, less compute and
fewer parameters. Buying context by widening a learned table was the expensive
way to solve a problem that was really about *how* position is represented.

This is the cleanest confirmation of Run 10's rule, and it was predicted in
advance rather than rationalised afterwards. A learned pos_emb must discover
what "position 400" means from gradient signal; each row is only trained when
sampled, and with 2,000 steps there is not enough signal to learn 512 of them
well. RoPE encodes relative position structurally — correct on step 1, and it
never spends a single step learning it. Amortised vs immediate predicts the
sign and roughly the size of every architecture result here: init scaling
(amortised, lost), learned positions (amortised, lost heavily), gating
(immediate, won), compression (immediate, won biggest).

Secondary benefit that matters at 98.7% of cap: RoPE refunds 81,920 params at
block 512 and decouples block_size from the parameter budget entirely, so
context becomes a pure compute decision.

---

## Run 11 — FINAL: RoPE + block 512 + RMSNorm + SwiGLU

**Hypothesis:** combine the three winners. Expected ~1.78 by naive addition of
their measured deltas, while flagging in advance that they would *not* simply
add: RoPE already fixes how position is represented, which is arguably part of
what widening the context was compensating for, so block 512's contribution
should shrink once RoPE does that job properly.

**What changed:** RoPE + block 512 + RMSNorm + SwiGLU together, on the Run 1
optimizer. Tokenizer BPE-4096 tied, as since Run 3.

**Result:** dev **bpb 1.7463**. **1,887,072 params (94.4% of cap), 2,000 steps.**
500 ms/step, 1001s. Sees 8,192,000 tokens = **390% of the corpus (3.9 epochs)**
against the baseline's 28% of one.

**vs baseline: 2.3718 -> 1.7463 = -0.6255 = -26.4%.**

**Conclusion:** Better than the -0.0721 the parts predicted from Run 9's 1.8184,
so the caution about non-additivity was right in direction but wrong in sign —
they reinforced rather than overlapped. The likely reason is that they relieve
*different* bottlenecks: RoPE fixes position representation, block 512 supplies
more of it to represent, and SwiGLU adds per-step expressiveness to use it. None
of the three is amortised; all three pay from step one, which is why they stack.

Worth stating what the final model is **not**: it is still 4 layers, 4 heads,
n_embd 160, init 0.05, dropout 0, batch 8 — every one of those is the baseline's
own value, because every attempt to change them lost. The entire -26.4% comes
from the tokenizer, the schedule, the positional scheme, the context width and
the MLP form. Nothing was made bigger; the same 2,000 steps were simply made to
read more and waste less.

---

## Vocabulary size — costed and rejected on the parameter cap

Not a training run: a measurement, made before spending 17 minutes on one.
Merges were trained at 1024 / 2048 / 4096 / 8192 (15-60s each) and each
vocabulary was costed against the final architecture and measured for
compression on dev_eval.txt:

| vocab | params @ final config | % of cap | bytes/token | context @ block 512 |
|------:|----------------------:|---------:|------------:|--------------------:|
| 1,024 | 1,395,552 | 69.8% | 2.690 | 1,377 bytes |
| 2,048 | 1,559,392 | 78.0% | 3.009 | 1,540 bytes |
| **4,096** | **1,887,072** | **94.4%** | **3.292** | **1,685 bytes** |
| 8,192 | 2,542,432 | **127.1%** | 3.582 | *(disqualified)* |

**Conclusion: 4,096 is the largest vocabulary the cap permits, and the cap is
what decides this.** 8,192 is the only option that compresses better than the
one shipped — 3.582 bytes/token, another ~9% of context — and it is 542,432
params over the limit at n_embd 160, which no other saving available here comes
close to closing. Every affordable alternative moves the *wrong way on the lever
that dominates this whole problem*: 2,048 gives up 8.6% of context and 1,024
gives up 18%, and both "buy" parameters this model cannot use, because the
binding constraint is steps, not capacity (Run 0, and every failed tuning run
since). So the vocabulary was not swept — the swept axis would have cost ~17
minutes per point to confirm a direction the cap and the compression numbers
already settle.

---

## Candidate levers (considered, not tested — no results claimed)

Nothing here is a result. Listed so the log shows what was considered and
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
