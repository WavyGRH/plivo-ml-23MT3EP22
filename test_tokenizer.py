"""Losslessness + compression checks for the BPE tokenizer.

evaluate.py refuses to score a tokenizer where decode(encode(text)) != text,
and the graders run that same round-trip on the hidden file. This checks the
property on the real corpora plus the edge cases most likely to break a
byte-level BPE: text the tokenizer never saw, scripts outside the training
mix, lone combining marks, and whitespace runs the pre-tokenizer must
reassemble exactly.

    python test_tokenizer.py
"""
import sys

import tokenizer as tokenizer_mod

CASES = {
    "empty": "",
    "single space": " ",
    "ascii": "The quick brown fox jumps over the lazy dog.",
    "devanagari": "यह एक परीक्षण वाक्य है।",
    "mixed": "In 2024, भारत ने 5G rollout किया — 100% coverage.",
    "whitespace runs": "a  b\t\tc\n\n\nd\r\ne   ",
    "leading/trailing ws": "   padded   ",
    "newline only": "\n",
    "emoji (unseen script)": "hello 👋🏽 world 🇮🇳",
    "cjk (unseen script)": "日本語のテキストです",
    "combining marks": "क़ ख़ ग़ ज़ ड़ ढ़ फ़ à é ñ",
    "control chars": "a\x00b\x01c\x7f",
    "zero-width": "a​b‌c‍d",
    "rtl": "مرحبا بالعالم",
    "digits/punct": "3.14159 == (2+2)*[7] {x} <y> #$%^&*",
    "repeated": "ab" * 500,
    "single byte": "\x01",
    "high codepoint": "\U0001F600\U0010FFFF",
}


def check(name, text, tok):
    try:
        ids = tok.encode(text)
        back = tok.decode(ids)
    except Exception as e:
        return False, f"raised {type(e).__name__}: {e}", 0
    if back != text:
        return False, "round-trip MISMATCH", len(ids)
    if ids and (min(ids) < 0 or max(ids) >= tok.vocab_size):
        return False, f"id out of range [0,{tok.vocab_size})", len(ids)
    return True, "ok", len(ids)


def main():
    tok = tokenizer_mod.load()
    print(f"tokenizer: {type(tok).__name__}, vocab_size {tok.vocab_size:,}\n")
    failures = 0

    for name, text in CASES.items():
        ok, msg, n = check(name, text, tok)
        flag = "PASS" if ok else "FAIL"
        failures += not ok
        print(f"  [{flag}] {name:<24} {n:>6} ids   {msg if not ok else ''}")

    print()
    for path in ("llm_handout/data/train_corpus.txt",
                 "llm_handout/data/dev_eval.txt"):
        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            print(f"  [SKIP] {path} not found")
            continue
        ok, msg, n = check(path, text, tok)
        failures += not ok
        n_bytes = len(text.encode("utf-8"))
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {path}")
        if ok:
            print(f"         {n_bytes:,} bytes -> {n:,} tokens "
                  f"= {n_bytes/n:.3f} bytes/token "
                  f"({n_bytes/n:.2f}x context vs raw bytes)")
        else:
            print(f"         {msg}")

    print()
    if failures:
        print(f"{failures} FAILURE(S) — do not ship this tokenizer.")
        sys.exit(1)
    print("All round-trips exact. Tokenizer is lossless.")


if __name__ == "__main__":
    main()
