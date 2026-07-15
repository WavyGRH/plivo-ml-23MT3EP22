"""Build SUMMARY.html from the actual artefacts, not from memory.

Reads RUNLOG.md, NOTES.md and ckpt.pt so every number on the page comes from a
file on disk. Regenerate any time with:  python make_summary.py
"""
import html
import json
import re
import subprocess

import torch

RUNS = []          # filled from RUNLOG.md
CKPT = "ckpt.pt"


def read_runlog():
    text = open("RUNLOG.md", encoding="utf-8").read()
    runs = []
    for block in re.split(r"\n---\n", text):
        m = re.search(r"^##\s+(.+)$", block, re.M)
        if not m:
            continue
        title = m.group(1).strip()
        bpb = re.search(r"dev \*\*bpb ([\d.]+)\*\*", block)
        hyp = re.search(r"\*\*Hypothesis:\*\*\s*(.+?)(?=\n\n|\Z)", block, re.S)
        chg = re.search(r"\*\*What changed:\*\*\s*(.+?)(?=\n\n|\Z)", block, re.S)
        con = re.search(r"\*\*Conclusion:\*\*\s*(.+?)(?=\n\n|\Z)", block, re.S)
        if bpb or "Run" in title or "Control" in title:
            runs.append({
                "title": title,
                "bpb": float(bpb.group(1)) if bpb else None,
                "hypothesis": clean(hyp.group(1)) if hyp else "",
                "changed": clean(chg.group(1)) if chg else "",
                "conclusion": clean(con.group(1)) if con else "",
            })
    return runs


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def ckpt_facts():
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = ck["model"]
    seen, dedup = set(), 0
    for v in sd.values():
        if v.data_ptr() in seen:
            continue
        seen.add(v.data_ptr())
        dedup += v.numel()
    return ck["config"], ck["steps"], dedup, ck.get("args", {})


def main():
    runs = read_runlog()
    cfg, steps, n_params, args = ckpt_facts()
    scored = [r for r in runs if r["bpb"] is not None]
    base = next((r["bpb"] for r in scored if "baseline" in r["title"].lower()), None)
    best = min((r["bpb"] for r in scored), default=None)
    notes = open("NOTES.md", encoding="utf-8").read() if __import__("os").path.exists("NOTES.md") else ""

    rows = []
    for r in scored:
        b = r["bpb"]
        delta = f"{b - base:+.4f}" if base and b != base else "—"
        cls = "win" if base and b < base else ("loss" if base and b > base else "")
        mark = " best" if b == best else ""
        rows.append(
            f'<tr class="{cls}{mark}"><td>{html.escape(r["title"])}</td>'
            f'<td class="num">{b:.4f}</td><td class="num">{delta}</td></tr>')

    cfg_rows = "".join(
        f"<tr><td><code>{html.escape(str(k))}</code></td>"
        f"<td class='num'>{html.escape(str(v))}</td></tr>"
        for k, v in sorted(cfg.items()))

    run_cards = "".join(
        f'<article class="{"loss" if base and r["bpb"] and r["bpb"] > base else "win"}">'
        f'<h3>{html.escape(r["title"])}'
        + (f' <span class="bpb">{r["bpb"]:.4f}</span>' if r["bpb"] else "")
        + "</h3>"
        + (f'<p><b>Hypothesis.</b> {r["hypothesis"]}</p>' if r["hypothesis"] else "")
        + (f'<p><b>Changed.</b> {r["changed"]}</p>' if r["changed"] else "")
        + (f'<p><b>Concluded.</b> {r["conclusion"]}</p>' if r["conclusion"] else "")
        + "</article>"
        for r in runs)

    improvement = f"{(best-base)/base*100:+.1f}%" if base and best else "—"
    label = {"H": "Human", "M": "Machine"}
    att_rows = "".join(
        f"<tr><td>{what}</td><td><b>{label[who]}</b></td></tr>"
        for what, who in ATTRIBUTION)

    doc = TEMPLATE.format(
        base=f"{base:.4f}" if base else "—",
        best=f"{best:.4f}" if best else "—",
        improvement=improvement,
        n_params=f"{n_params:,}",
        pct_cap=f"{n_params/2_000_000*100:.1f}",
        steps=steps,
        rows="".join(rows),
        cfg_rows=cfg_rows,
        run_cards=run_cards,
        notes=html.escape(notes),
        n_runs=len(scored),
        attribution=att_rows,
        att_summary=ATT_SUMMARY,
    )
    open("SUMMARY.html", "w", encoding="utf-8").write(doc)
    print(f"wrote SUMMARY.html  ({len(doc):,} bytes, {len(scored)} scored runs)")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2,000 Step LLM Speedrun — SUMMARY</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --mut:#666; --line:#e3e3e3;
           --win:#0a7d33; --loss:#c0392b; --card:#fafafa; }}
  @media (prefers-color-scheme:dark) {{
    :root {{ --bg:#141414; --fg:#e8e8e8; --mut:#999; --line:#2e2e2e;
             --win:#4ade80; --loss:#f87171; --card:#1c1c1c; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0 auto; padding:2rem 1.25rem 5rem; max-width:52rem;
         background:var(--bg); color:var(--fg);
         font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  h1 {{ font-size:1.9rem; margin:0 0 .25rem; letter-spacing:-.02em; }}
  h2 {{ font-size:1.25rem; margin:2.75rem 0 .85rem; padding-bottom:.4rem;
        border-bottom:1px solid var(--line); letter-spacing:-.01em; }}
  h3 {{ font-size:1rem; margin:0 0 .5rem; }}
  .sub {{ color:var(--mut); margin:0 0 2rem; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88em;
          background:var(--card); padding:.1em .35em; border-radius:3px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(8.5rem,1fr));
           gap:.75rem; margin:1.5rem 0; }}
  .kpi {{ background:var(--card); border:1px solid var(--line);
          border-radius:8px; padding:.9rem 1rem; }}
  .kpi .v {{ font-size:1.5rem; font-weight:640; letter-spacing:-.02em;
             font-variant-numeric:tabular-nums; }}
  .kpi .l {{ color:var(--mut); font-size:.75rem; text-transform:uppercase;
             letter-spacing:.06em; margin-top:.15rem; }}
  .scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }}
  table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
  th,td {{ text-align:left; padding:.5rem .65rem; border-bottom:1px solid var(--line); }}
  th {{ color:var(--mut); font-weight:600; font-size:.75rem;
        text-transform:uppercase; letter-spacing:.05em; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  tr.win td:nth-child(3) {{ color:var(--win); }}
  tr.loss td:nth-child(3) {{ color:var(--loss); }}
  tr.best td {{ font-weight:700; }}
  article {{ background:var(--card); border:1px solid var(--line);
             border-left:3px solid var(--win); border-radius:0 8px 8px 0;
             padding:.9rem 1.1rem; margin:.75rem 0; }}
  article.loss {{ border-left-color:var(--loss); }}
  article p {{ margin:.4rem 0; font-size:.9rem; }}
  article b {{ color:var(--mut); font-weight:600; }}
  .bpb {{ float:right; font-family:ui-monospace,monospace; font-weight:700; }}
  pre {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
         padding:1rem; overflow-x:auto; font-size:.85rem; white-space:pre-wrap; }}
  .split {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }}
  @media (max-width:640px) {{ .split {{ grid-template-columns:1fr; }} }}
  .att th:first-child {{ width:40%; }}
</style></head><body>

<h1>2,000 Step LLM Speedrun</h1>
<p class="sub">Bits per byte on held-out text — lower is better.
Caps: 2,000 optimizer steps, 2,000,000 parameters, CPU only,
train_corpus.txt only, pure PyTorch.</p>

<div class="kpis">
  <div class="kpi"><div class="v">{base}</div><div class="l">baseline bpb</div></div>
  <div class="kpi"><div class="v">{best}</div><div class="l">final bpb</div></div>
  <div class="kpi"><div class="v">{improvement}</div><div class="l">improvement</div></div>
  <div class="kpi"><div class="v">{n_params}</div><div class="l">params ({pct_cap}% of cap)</div></div>
  <div class="kpi"><div class="v">{steps}</div><div class="l">steps (cap 2000)</div></div>
</div>

<h2>Every scored run</h2>
<div class="scroll"><table>
<thead><tr><th>Run</th><th class="num">dev bpb</th><th class="num">vs baseline</th></tr></thead>
<tbody>{rows}</tbody></table></div>

<h2>Final architecture</h2>
<div class="scroll"><table>
<thead><tr><th>Config key</th><th class="num">Value</th></tr></thead>
<tbody>{cfg_rows}</tbody></table></div>

<h2>NOTES.md</h2>
<pre>{notes}</pre>

<h2>Run-by-run reasoning ({n_runs} scored runs)</h2>
{run_cards}

<h2>Machine-done vs human-done</h2>
<p>The brief permits AI coding assistants and requires this section to be
honest, so it is written to be accurate rather than flattering. This work used
Claude Code (Opus 4.8) throughout.</p>
<div class="scroll"><table class="att">
<thead><tr><th>Contribution</th><th>By</th></tr></thead>
<tbody>{attribution}</tbody></table></div>
<p><b>Summary in one line:</b> {att_summary}</p>

</body></html>
"""


# Honest attribution. H = Devangan (human), M = Claude Code (machine).
ATTRIBUTION = [
    ("Track selection; environment setup and verification; repo and remote", "H"),
    ("Deadline discipline and submission constraints; instruction not to "
     "violate any evaluation criteria", "H"),
    ("Strategic direction: asked for the option space, then pushed for more "
     "ambition ('look for more ways') rather than accepting safe tuning", "H"),
    ("Chose the working split (human picks, machine executes) and set "
     "priorities between runs", "H"),
    ("Questions that redirected the work: whether handout files must be "
     "pushed; whether the 2,000-step cap was being honoured; the RMSNorm + "
     "tying cost analysis; and the request to verify committed work against "
     "the caps &mdash; which is what surfaced the tied-weight "
     "double-counting risk", "H"),
    ("Reading the brief; extracting caps, frozen interfaces and deliverables; "
     "corpus measurement (Devanagari share, bytes/token, corpus coverage)", "M"),
    ("All code: BPE tokenizer and its training, losslessness test suite, "
     "RoPE / RMSNorm / SwiGLU / decoupled-QK attention, trainer refactor, "
     "this summary generator", "M"),
    ("All hypotheses, experiment design, and run execution", "M"),
    ("All RUNLOG.md analysis and conclusions, including diagnosing and "
     "correcting its own falsified explanation of Run 2", "M"),
    ("Parameter-cap arithmetic; compliance auditing; catching two "
     "self-inflicted bugs (overwriting modules mid-run; queueing runs on a "
     "known-bad config)", "M"),
]

ATT_SUMMARY = (
    "Substantially machine-executed. Every line of code, every hypothesis, and "
    "every RUNLOG conclusion was produced by Claude Code. The human "
    "contribution was direction rather than implementation: choosing the "
    "track and the working method, enforcing the caps and the deadline, "
    "pushing for ambitious swings over safe tuning, and asking the "
    "verification questions that caught a disqualification-level risk. "
    "Presenting this as primarily human work would be false."
)

if __name__ == "__main__":
    main()
