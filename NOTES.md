# NOTES — best configuration and why it works

*(Brief caps this at 10 sentences; RUNLOG.md has every run, including failures.)*

1. **Best configuration — dev bpb 1.7463 vs the baseline's 2.3718 (−26.4%), at
   1,887,072 parameters (94.4% of the cap) and exactly 2,000 steps:** byte-level
   BPE trained on the corpus alone (vocab 4,096, 3.49 bytes/token) with the
   output head tied to the token embedding, RoPE, block 512, RMSNorm, SwiGLU at
   hidden = 8/3·d, over the baseline's own 4 layers × 4 heads × n_embd 160,
   trained with AdamW at peak lr 1e-3, 100-step warmup and cosine decay to zero,
   weight decay 0.1 on matrices only, gradient clipping 1.0, β₂ 0.95,
   init N(0, 0.05), dropout 0.
2. Everything follows from one measurement on the baseline: its loss is still
   falling steeply at step 2,000 and it reads only 28% of its corpus, so the
   binding constraint is the **step cap**, never model capacity.
3. **That yields the rule which predicted every architecture result here:** under
   a hard step cap, a technique wins if its benefit is *immediate* and loses if
   it is *amortised over steps*, because a 2,000-step budget never collects the
   latter.
4. **The tokenizer is the largest lever (−0.2014):** the corpus is 14.1%
   Devanagari and the eval text 20.5%, and UTF-8 spends three bytes per
   Devanagari character, so a byte tokenizer burns its context precisely where
   the evaluation is densest — BPE turns 28% corpus coverage into 3.9 epochs and
   widens context from 128 bytes to ~1,685.
5. **Weight tying is not an optimisation, it is what makes BPE legal:** untied,
   vocab 4,096 costs 2,589,120 parameters (129% of cap); tied, 1,933,760 — so the
   starter's `tie_weights = False` is what makes the biggest win look impossible.
6. **RoPE is the second largest (−0.1740) and the rule's cleanest case:** a
   learned `pos_emb` must *learn* what each position means before it is useful
   (amortised), while rotary is correct on step one — RoPE at block 256 beats
   learned positions at block 512, with fewer parameters and less compute.
7. **A schedule matters more here than in a normal run (−0.0983):** with no
   option to train longer, the learning rate must complete its whole job inside
   2,000 steps, so the baseline's defect was never the value 3e-4 but that it
   never changed.
8. Context width paid twice (128→256: −0.0789; 256→512: −0.0342) into clear
   diminishing returns, and RMSNorm+SwiGLU won narrowly (−0.0066) while
   param-matched *downward* — so that gain is the gating, not extra capacity.
9. **Every attempt to tune a baseline constant failed** (init 0.02 + residual
   scaling: +0.153; init 0.08: +0.070; lr 2e-3: +0.008), so the shipped model is
   still architecturally the baseline's — same depth, width, heads, init,
   dropout and batch size.
10. Nothing here was made bigger: the same 2,000 steps were simply made to read
    3.9 epochs instead of 28% of one, and to stop wasting them learning things
    (position, weight scale) that could be had for free.
