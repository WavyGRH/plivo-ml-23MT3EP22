"""Byte-level BPE trained on train_corpus.txt only.

Why not raw bytes: the corpus is 14% Devanagari and the eval text is ~20%.
UTF-8 spends 3 bytes per Devanagari codepoint, so a byte tokenizer burns three
token slots per Hindi character — shortening the model's effective context
exactly where the eval is densest, and capping it at 128 bytes of history.

Losslessness (evaluate.py hard-fails without it, and so do the graders):
the base vocabulary is all 256 byte values, so *any* byte sequence is
representable and merges only ever replace a pair with a token that decodes
back to the same two pieces. decode(encode(text)) == text by construction,
for arbitrary UTF-8 — that is the byte fallback.

Interface kept as required: load() takes no arguments and returns an object
with .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
Merges are read from bpe_merges.json resolved relative to __file__, so grading
works with cwd set anywhere and with no internet.

Retrain with:  python tokenizer.py --data llm_handout/data/train_corpus.txt --vocab 4096
"""
import argparse
import json
import os
import re
import time
from collections import Counter

MERGES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bpe_merges.json")

# Total coverage of any string: every character is a word char, whitespace, or
# neither, so re.findall reconstructs the input exactly (asserted in _chunks).
# The optional leading space follows GPT-2: " the" is one token, not two.
_PAT = re.compile(r" ?\w+| ?[^\w\s]+|\s+")


def _chunks(text):
    out = _PAT.findall(text)
    assert "".join(out) == text, "pre-tokenizer is lossy"
    return out


class BPETokenizer:
    def __init__(self, merges):
        # merges: list of [a, b] pairs, in learned order
        self.merges = [tuple(m) for m in merges]
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)
        self.vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache = {}

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # merge the pair with the lowest rank (learned earliest)
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new_id = 256 + best_rank
            merged, i = [], 0
            while i < len(ids):
                if (i < len(ids) - 1 and ids[i] == best[0]
                        and ids[i + 1] == best[1]):
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in _chunks(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        buf = b"".join(self.vocab[i] for i in ids)
        return buf.decode("utf-8", errors="replace")


def train(data_path, vocab_size, out_path=MERGES_FILE, verbose=True):
    """Learn merges from the corpus. Counts are kept per unique chunk and
    updated incrementally, so cost scales with vocabulary churn rather than
    with corpus length."""
    text = open(data_path, encoding="utf-8").read()
    t0 = time.time()
    freqs = Counter(_chunks(text))
    words = [list(w.encode("utf-8")) for w in freqs]
    counts = list(freqs.values())
    if verbose:
        print(f"{len(text):,} chars -> {len(words):,} unique chunks")

    pair_counts = Counter()
    where = {}                       # pair -> set of word indices
    for wi, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += counts[wi]
            where.setdefault(pair, set()).add(wi)

    merges = []
    target = vocab_size - 256
    while len(merges) < target:
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        if pair_counts[best] < 2:
            break
        new_id = 256 + len(merges)
        merges.append(list(best))

        for wi in list(where.get(best, ())):
            w = words[wi]
            c = counts[wi]
            # drop this word's old pair contributions
            for pair in zip(w, w[1:]):
                pair_counts[pair] -= c
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                s = where.get(pair)
                if s is not None:
                    s.discard(wi)
            nw, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                    nw.append(new_id)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            words[wi] = nw
            for pair in zip(nw, nw[1:]):
                pair_counts[pair] += c
                where.setdefault(pair, set()).add(wi)

        if verbose and len(merges) % 500 == 0:
            print(f"  {len(merges):5d}/{target} merges  ({time.time()-t0:.0f}s)")

    with open(out_path, "w") as f:
        json.dump({"merges": merges}, f)
    if verbose:
        print(f"wrote {out_path}: vocab {256+len(merges):,} "
              f"({time.time()-t0:.0f}s)")
    return BPETokenizer(merges)


class ByteTokenizer:
    """The original baseline: raw UTF-8 bytes, vocab 256. Kept so RUNLOG's
    byte-tokenizer control runs stay reproducible after BPE landed."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def load(path=None):
    """Return the tokenizer used by train.py and evaluate.py. No arguments.

    Grading calls this with no arguments and no environment set, and gets the
    BPE tokenizer the checkpoint was trained with. PLIVO_TOK=byte exists only
    to reproduce the baseline control runs recorded in RUNLOG.md.
    """
    if os.environ.get("PLIVO_TOK") == "byte":
        return ByteTokenizer()
    # PLIVO_MERGES likewise exists only for the vocab-size sweep in RUNLOG.md;
    # unset (as at grading) it resolves to bpe_merges.json next to this file.
    path = path or os.environ.get("PLIVO_MERGES") or MERGES_FILE
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} missing — run: python tokenizer.py --data "
            f"llm_handout/data/train_corpus.txt --vocab 4096")
    with open(path) as f:
        return BPETokenizer(json.load(f)["merges"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--out", default=MERGES_FILE)
    a = ap.parse_args()
    train(a.data, a.vocab, a.out)
