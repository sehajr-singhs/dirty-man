"""Build docs/playground.html from results/playground.json.

Embeds the routing data directly in the HTML so the page works from file://
with no server and no fetch. Run: python make_playground.py
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "results", "playground.json")
OUT = os.path.join(ROOT, "docs", "playground.html")

with open(SRC) as f:
    data = json.load(f)

prims = data["primitives"]
severities = data["severity"]
labels = data["label"]
images = data["images"]
probs = data["probs"]

# Round routing probs for embedding (keep enough precision for the bars).
probs_r = [[round(p, 6) for p in row] for row in probs]

payload = json.dumps({
    "primitives": prims,
    "severity": severities,
    "label": labels,
    "images": images,
    "probs": probs_r,
})

N = len(images)

html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Switch Operator — live playground</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/computer-modern/cmu-serif.css">
  <style>
    :root { --ink:#1a1a1a; --muted:#555; --faint:#8c8e90; --panel:#f8f8f8; --border:#c4c6c8; --link:#226999; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family:'CMU Serif', Georgia, serif; color: var(--ink); background:#fff; -webkit-font-smoothing: antialiased; }
    .container { max-width: 1080px; margin: 0 auto; padding: 0 20px; }
    .hero { padding: 3rem 0 1.2rem; text-align: center; }
    .hero h1 { font-size: 2.2rem; font-weight: 900; line-height: 1.15; }
    .hero p { color: var(--muted); margin-top: 0.6rem; font-size: 1.05rem; }
    .hero a { color: var(--link); }
    .controls { text-align: center; margin: 1.4rem 0; }
    .controls button {
      font-family:'IBM Plex Mono', monospace; font-size: 0.8rem;
      border: 1px solid var(--border); background: var(--panel); color: var(--ink);
      border-radius: 8px; padding: 0.5em 1.1em; cursor: pointer; margin: 0 0.25rem;
    }
    .controls button:hover { background:#eee; }
    .stats { font-family:'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--faint);
             text-align: center; margin-bottom: 1.4rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 1rem; }
    .card { border: 1px solid #e4e4e4; border-radius: 10px; overflow: hidden; background: #fff; }
    .card-head { display: flex; align-items: center; gap: 0.9rem; padding: 0.8rem 0.9rem;
                 border-bottom: 1px solid #eee; }
    .card-head canvas { border: 1px solid #ddd; border-radius: 6px; background: #fff; }
    .card-head .meta { flex: 1; }
    .card-head .meta .label { font-weight: 700; font-size: 0.95rem; }
    .card-head .meta .sev { font-family:'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--faint); }
    .bars { padding: 0.7rem 0.9rem 0.9rem; }
    .bar-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.18rem 0; }
    .bar-row .pname { width: 84px; font-family:'IBM Plex Mono', monospace; font-size: 0.66rem;
                      color: var(--muted); text-align: right; }
    .bar-track { flex: 1; height: 9px; background: #f0f0f0; border-radius: 5px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--link); border-radius: 5px; width: 0%; }
    .bar-row.win .bar-fill { background: #1a7a3c; }
    .bar-row.win .pname { color: var(--ink); font-weight: 700; }
    .bar-row .pct { width: 44px; font-family:'IBM Plex Mono', monospace; font-size: 0.64rem; color: var(--faint); }
    .footer { margin-top: 2.4rem; padding: 1.6rem 0 3rem; border-top: 1px solid #e6e6e6;
              text-align: center; color: var(--faint); font-size: 0.85rem; }
    .footer a { color: var(--link); }
    .legend { font-family:'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--muted);
              text-align: center; margin-bottom: 1rem; }
    .legend b { color: var(--ink); }
  </style>
</head>
<body>
<div class="container">
  <div class="hero">
    <h1>The Switch Operator — live routing</h1>
    <p>What the operator <em>chose</em>, sample by sample, on real (corrupted) glyphs.
      Each card is one input: the glyph on the left, the router's primitive probabilities on the right.
      Sort by corruption severity and watch flat lenses give way to spatial lenses. <a href="index.html">Back to the project page</a></p>
  </div>
  <div class="controls">
    <button id="btn-sev" class="active">sort by severity</button>
    <button id="btn-label">sort by digit</button>
    <button id="btn-lens">sort by chosen lens</button>
  </div>
  <div class="stats" id="stats"></div>
  <div class="legend"><b>green</b> = chosen primitive &middot; severity 0 = clean sim &rarr; 1 = dirtiest real</div>
  <div class="grid" id="grid"></div>
  <div class="footer"><p><a href="index.html">project page</a> &middot; <a href="papers/nmi_paper.pdf">paper</a> &middot;
    <a href="papers/ieee_paper.pdf">IEEE paper</a> &middot; data from <code>results/playground.json</code></p></div>
</div>

<script>
const DATA = __PAYLOAD__;
const prims = DATA.primitives;
const n = DATA.images.length;

function renderGlyph(canvas, img) {
  const ctx = canvas.getContext('2d');
  const size = 72;
  canvas.width = size; canvas.height = size;
  const imgData = ctx.createImageData(size, size);
  const flat = img[0];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const sx = Math.min(23, Math.floor(x * 24 / size));
      const sy = Math.min(23, Math.floor(y * 24 / size));
      const v = Math.round(flat[sy][sx] * 255);
      const i = (y * size + x) * 4;
      imgData.data[i] = v; imgData.data[i+1] = v; imgData.data[i+2] = v; imgData.data[i+3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

function buildCard(idx) {
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.idx = idx;

  const head = document.createElement('div');
  head.className = 'card-head';
  const cv = document.createElement('canvas');
  head.appendChild(cv);
  const meta = document.createElement('div');
  meta.className = 'meta';
  const lab = document.createElement('div');
  lab.className = 'label';
  lab.textContent = 'digit ' + DATA.label[idx];
  const sev = document.createElement('div');
  sev.className = 'sev';
  sev.textContent = 'severity ' + DATA.severity[idx].toFixed(2) +
    (DATA.severity[idx] < 0.001 ? '  (clean sim)' : '');
  meta.appendChild(lab); meta.appendChild(sev);
  head.appendChild(meta);
  card.appendChild(head);

  const bars = document.createElement('div');
  bars.className = 'bars';
  const p = DATA.probs[idx];
  const best = p.indexOf(Math.max(...p));
  for (let k = 0; k < prims.length; k++) {
    const row = document.createElement('div');
    row.className = 'bar-row' + (k === best ? ' win' : '');
    const nm = document.createElement('div');
    nm.className = 'pname'; nm.textContent = prims[k];
    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.width = (p[k] * 100).toFixed(1) + '%';
    track.appendChild(fill);
    const pct = document.createElement('div');
    pct.className = 'pct';
    pct.textContent = (p[k] * 100).toFixed(1) + '%';
    row.appendChild(nm); row.appendChild(track); row.appendChild(pct);
    bars.appendChild(row);
  }
  card.appendChild(bars);
  return card;
}

const grid = document.getElementById('grid');
const cards = [];
for (let i = 0; i < n; i++) {
  const c = buildCard(i);
  cards.push(c);
  grid.appendChild(c);
  renderGlyph(c.querySelector('canvas'), DATA.images[i]);
}

function sortBy(key) {
  const order = Array.from({length: n}, (_, i) => i);
  if (key === 'sev') order.sort((a, b) => DATA.severity[a] - DATA.severity[b]);
  else if (key === 'label') order.sort((a, b) => DATA.label[a] - DATA.label[b] || DATA.severity[a] - DATA.severity[b]);
  else {
    order.sort((a, b) => {
      const ba = DATA.probs[a].indexOf(Math.max(...DATA.probs[a]));
      const bb = DATA.probs[b].indexOf(Math.max(...DATA.probs[b]));
      return ba - bb || DATA.severity[a] - DATA.severity[b];
    });
  }
  order.forEach((idx, pos) => grid.appendChild(cards[idx]));
  const chosen = {};
  for (const i of order) {
    const b = DATA.probs[i].indexOf(Math.max(...DATA.probs[i]));
    chosen[prims[b]] = (chosen[prims[b]] || 0) + 1;
  }
  const total = order.length;
  document.getElementById('stats').textContent =
    'samples: ' + total + '  ·  chosen lenses: ' +
    Object.entries(chosen).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => k + ' ' + v).join('  ·  ');
}

document.getElementById('btn-sev').onclick = () => { setActive('btn-sev'); sortBy('sev'); };
document.getElementById('btn-label').onclick = () => { setActive('btn-label'); sortBy('label'); };
document.getElementById('btn-lens').onclick = () => { setActive('btn-lens'); sortBy('lens'); };

function setActive(id) {
  for (const b of document.querySelectorAll('.controls button')) b.classList.remove('active');
  document.getElementById(id).classList.add('active');
}

sortBy('sev');
</script>
</body>
</html>
"""

html = html.replace("__PAYLOAD__", payload)

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({N} samples)")
