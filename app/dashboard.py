"""Interactive dashboard — FastAPI + Plotly.

Run: uvicorn app.dashboard:app --port 8000
"""
import os
import json
import pathlib
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from model.calibration import Parameters, h_sigma
from model.analytical import simulate_forward, find_optimal_alpha
from model.policy import compute_regime_comparison, compute_delay_cost_curve, score_domain, DEFAULT_DOMAINS
from model.evolutionary import simulate_evolution

app = FastAPI(title="Sovereignty Model")

STATIC_DIR = pathlib.Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

COLORS = {
    "drift": "#b0a896", "destructive": "#c4756a", "designed": "#5a8a6a",
    "present": "#8B4049", "future": "#5a7a8a", "dependency": "#b89a3c",
}


@app.get("/api/simulate")
def api_simulate(
    alpha: float = Query(0.5, ge=0, le=1),
    sigma: float = Query(2.0, ge=0, le=20),
    D0: float = Query(0.67, ge=0, le=1),
    R: float = Query(0.022, ge=0.001, le=0.1),
    T: float = Query(30, ge=5, le=50),
    gamma: float = Query(0.1, ge=0.01, le=0.5),
    rho: float = Query(0.03, ge=0.01, le=0.15),
):
    params = Parameters(D0=D0, R=R, T=T, gamma=gamma, rho=rho)
    result = simulate_forward(alpha, sigma, params)
    return {
        "t": result.t.tolist(),
        "Kp": result.Kp.tolist(),
        "Kf": result.Kf.tolist(),
        "D": result.D.tolist(),
        "V_total": float(result.V_total),
    }


@app.get("/api/regimes")
def api_regimes(
    D0: float = Query(0.67), R: float = Query(0.022),
    T: float = Query(30), gamma: float = Query(0.1), rho: float = Query(0.03),
):
    params = Parameters(D0=D0, R=R, T=T, gamma=gamma, rho=rho)
    regimes = compute_regime_comparison(params)
    return {
        name: {
            "t": r.t.tolist(), "Kp": r.Kp.tolist(), "Kf": r.Kf.tolist(),
            "D": r.D.tolist(), "V_total": float(r.V_total),
        }
        for name, r in regimes.items()
    }


@app.get("/api/delay")
def api_delay(
    alpha: float = Query(0.3), sigma: float = Query(2.0),
    max_delay: float = Query(15.0),
    D0: float = Query(0.67), R: float = Query(0.022),
    T: float = Query(30), gamma: float = Query(0.1), rho: float = Query(0.03),
):
    params = Parameters(D0=D0, R=R, T=T, gamma=gamma, rho=rho)
    delays, costs = compute_delay_cost_curve(alpha, sigma, max_delay, params)
    return {"delays": delays.tolist(), "costs": costs.tolist()}


@app.get("/api/evolutionary")
def api_evolutionary(
    sigma: float = Query(2.0), N: int = Query(200, ge=50, le=500),
    T: float = Query(30), seed: int = Query(42),
):
    params = Parameters(N=N, T=T)
    result = simulate_evolution(sigma, params, seed=seed)
    return {
        "t": result.t.tolist(),
        "aggregate_capability": result.aggregate_capability.tolist(),
        "avg_dependency": result.avg_dependency.tolist(),
    }


@app.get("/api/hump")
def api_hump():
    sigmas = np.linspace(0, 15, 200).tolist()
    h_vals = [float(h_sigma(s)) for s in sigmas]
    return {"sigmas": sigmas, "h_values": h_vals, "sigma_star": 2.0}


@app.get("/api/domains")
def api_domains():
    results = []
    for name, assessment in DEFAULT_DOMAINS.items():
        s = score_domain(**assessment)
        results.append({"name": name.replace("_", " ").title(), "score": s, **assessment})
    return sorted(results, key=lambda x: x["score"], reverse=True)


@app.get("/pdf")
def serve_pdf():
    pdf_path = STATIC_DIR / "sovereignty_model.pdf"
    if pdf_path.exists():
        return FileResponse(pdf_path, media_type="application/pdf", filename="sovereignty_model.pdf")
    return {"error": "PDF not found"}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Two Kinds of Sovereignty &mdash; Economic Model</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,600&family=Inter:wght@400;500;600&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Source Serif 4', Georgia, serif; background: #FAFAF7; color: #1a1a1a; line-height: 1.6; }
  .page { max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.5rem; }
  h1 { font-family: 'Playfair Display', serif; font-size: 1.8rem; margin-bottom: 0.25rem; }
  h2 { font-family: 'Playfair Display', serif; font-size: 1.3rem; margin: 2.5rem 0 0.5rem; color: #1a1a1a; }
  h3 { font-family: 'Playfair Display', serif; font-size: 1.05rem; margin: 1.5rem 0 0.3rem; font-weight: 600; }
  .subtitle { font-size: 0.95rem; color: #777; font-style: italic; margin-bottom: 0.75rem; }
  .pdf-link { display: inline-block; font-family: 'Inter', sans-serif; font-size: 0.8rem; background: #5a8a6a; color: white; padding: 0.4rem 1rem; border-radius: 4px; text-decoration: none; margin: 0.75rem 0; }
  .pdf-link:hover { background: #4a7a5a; }

  /* Prose blocks for explanations */
  .prose { font-size: 0.92rem; color: #444; max-width: 720px; margin-bottom: 1rem; }
  .prose p { margin-bottom: 0.7rem; }
  .prose strong { color: #1a1a1a; }

  /* Callout boxes */
  .callout { font-family: 'Inter', sans-serif; font-size: 0.8rem; padding: 0.8rem 1rem; border-radius: 5px; margin: 1rem 0; line-height: 1.5; }
  .callout-insight { background: #e5f0e8; border-left: 3px solid #5a8a6a; color: #2d5a3a; }
  .callout-warning { background: #fce8e5; border-left: 3px solid #c4756a; color: #7a3830; }
  .callout-neutral { background: #f5f3ee; border-left: 3px solid #b0a896; color: #5a5347; }
  .callout-label { font-weight: 600; text-transform: uppercase; font-size: 0.65rem; letter-spacing: 0.06em; margin-bottom: 0.3rem; }

  /* Slider explanation labels */
  .slider-explain { font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #999; margin-top: 0.15rem; line-height: 1.3; max-width: 160px; }

  .tabs { display: flex; gap: 0; border-bottom: 2px solid #e0ddd6; margin: 1.5rem 0 0; }
  .tab { font-family: 'Inter', sans-serif; font-size: 0.85rem; padding: 0.6rem 1.2rem; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -2px; color: #888; transition: all 0.2s; }
  .tab:hover { color: #555; }
  .tab.active { color: #1a1a1a; border-bottom-color: #5a8a6a; font-weight: 600; }
  .tab-content { display: none; padding: 1.5rem 0; }
  .tab-content.active { display: block; }
  .controls { display: flex; flex-wrap: wrap; gap: 1.25rem; margin: 1rem 0; padding: 1rem; background: #f5f3ee; border-radius: 6px; }
  .control { display: flex; flex-direction: column; gap: 0.25rem; }
  .control label { font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 500; color: #666; }
  .control input[type=range] { width: 160px; accent-color: #5a8a6a; }
  .control .val { font-family: 'Inter', sans-serif; font-size: 0.75rem; color: #5a8a6a; font-weight: 600; }
  .metric-row { display: flex; gap: 1.5rem; margin: 1rem 0; }
  .metric { font-family: 'Inter', sans-serif; padding: 0.75rem 1rem; background: #f5f3ee; border-radius: 4px; }
  .metric .label { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric .value { font-size: 1.1rem; font-weight: 600; margin-top: 0.2rem; }
  .chart { margin: 1rem 0; }
  .chart-caption { font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #999; margin-top: 0.3rem; line-height: 1.4; }
  .domain-bar { margin: 0.75rem 0; }
  .domain-bar .bar-label { font-family: 'Inter', sans-serif; font-size: 0.8rem; margin-bottom: 0.2rem; }
  .domain-bar .bar-track { height: 28px; background: #f0ece4; border-radius: 3px; overflow: hidden; }
  .domain-bar .bar-fill { height: 100%; border-radius: 3px; background: #5a8a6a; display: flex; align-items: center; padding-left: 0.5rem; }
  .domain-bar .bar-fill span { font-family: 'Inter', sans-serif; font-size: 0.7rem; color: white; font-weight: 600; }
  .domain-bar .bar-detail { font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #888; margin-top: 0.2rem; }
  .regime-metrics { display: flex; gap: 1rem; margin: 1rem 0; }
  .regime-metric { flex: 1; text-align: center; padding: 0.75rem; border-radius: 4px; }
  .regime-metric .rlabel { font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #666; }
  .regime-metric .rdesc { font-family: 'Inter', sans-serif; font-size: 0.65rem; color: #999; margin-top: 0.15rem; }
  .regime-metric .rvalue { font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 600; margin-top: 0.2rem; }
  button.run-btn { font-family: 'Inter', sans-serif; font-size: 0.8rem; padding: 0.5rem 1rem; background: #5a8a6a; color: white; border: none; border-radius: 4px; cursor: pointer; }
  button.run-btn:hover { background: #4a7a5a; }
  .section-divider { border: none; border-top: 1px solid #e8e6e1; margin: 2rem 0; }

  /* How to read guide */
  .read-guide { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.75rem 0; }
  .read-guide-item { font-family: 'Inter', sans-serif; font-size: 0.72rem; display: flex; align-items: center; gap: 0.4rem; }
  .guide-swatch { width: 20px; height: 3px; border-radius: 2px; }
</style>
</head>
<body>
<div class="page">
  <h1>Two Kinds of Sovereignty &mdash; Economic Model</h1>
  <p class="subtitle">An interactive companion to &ldquo;Two Kinds of Sovereignty.&rdquo; This tool lets you explore the economic model behind the essay&rsquo;s argument: that Europe must shift resources from securing today&rsquo;s technology toward building tomorrow&rsquo;s.</p>

  <div class="prose">
    <p>The essay argues that sovereignty comes in two forms. <strong>Present sovereignty</strong> means controlling and securing the technology you already depend on &mdash; keeping the lights on. <strong>Future sovereignty</strong> means investing in new technological domains before the window of opportunity closes. The central question is how to split limited resources between these two goals.</p>
    <p>This model makes that tradeoff precise. Use the three tabs below to explore different layers of the argument, from the big-picture allocation problem down to what happens at the level of individual firms. <a href="/pdf" style="color:#5a8a6a;">The full formal specification is available as a PDF.</a></p>
  </div>

  <a href="/pdf" class="pdf-link">Download formal model specification (PDF)</a>

  <div class="tabs" id="tab-bar"></div>

  <!-- ============================================================ -->
  <!-- TAB 1: ANALYTICAL CORE -->
  <!-- ============================================================ -->
  <div id="analytical" class="tab-content active">
    <h2>The Allocation Problem</h2>
    <div class="prose">
      <p>Imagine you are a European policymaker with a fixed R&amp;D budget. You must decide: how much goes to <strong>shoring up today&rsquo;s systems</strong> (cloud resilience, regulatory capacity, domesticating existing technology) versus <strong>investing in the next paradigm</strong> (scientific AI, quantum computing, fusion energy)?</p>
      <p>This view shows what happens over 30 years under different choices. The chart tracks three things simultaneously:</p>
    </div>

    <div class="read-guide">
      <div class="read-guide-item"><div class="guide-swatch" style="background:#8B4049;"></div> <strong>Present sovereignty</strong> &mdash; your control over today&rsquo;s tech stack</div>
      <div class="read-guide-item"><div class="guide-swatch" style="background:#5a7a8a;"></div> <strong>Future sovereignty</strong> &mdash; your capacity in emerging domains</div>
      <div class="read-guide-item"><div class="guide-swatch" style="background:#b89a3c;"></div> <strong>Dependency</strong> &mdash; how locked-in you are to foreign providers (right axis, 0&ndash;100%)</div>
    </div>

    <div class="controls">
      <div class="control">
        <label>How much goes to today&rsquo;s systems?</label>
        <input type="range" id="alpha" min="0" max="1" step="0.01" value="0.5" oninput="updateAnalytical()">
        <span class="val" id="alpha-val">0.50</span>
        <div class="slider-explain">0 = everything to future, 1 = everything to present. Default 0.67 reflects Europe&rsquo;s current tilt.</div>
      </div>
      <div class="control">
        <label>How strong is the &ldquo;designed constraint&rdquo;?</label>
        <input type="range" id="sigma" min="0" max="15" step="0.1" value="2.0" oninput="updateAnalytical()">
        <span class="val" id="sigma-val">2.0</span>
        <div class="slider-explain">The essay&rsquo;s &ldquo;virtual export restriction.&rdquo; Mild pressure spurs innovation; too much destroys it.</div>
      </div>
      <div class="control">
        <label>Starting dependency level</label>
        <input type="range" id="D0" min="0" max="1" step="0.01" value="0.67" oninput="updateAnalytical()">
        <span class="val" id="D0-val">0.67</span>
        <div class="slider-explain">Europe starts at ~67% (US hyperscaler cloud share). Drag lower to see a less dependent starting point.</div>
      </div>
      <div class="control">
        <label>How fast do opportunity windows close?</label>
        <input type="range" id="gamma" min="0.01" max="0.3" step="0.01" value="0.1" oninput="updateAnalytical()">
        <span class="val" id="gamma-val">0.10</span>
        <div class="slider-explain">Higher = windows close faster, making delayed investment more costly.</div>
      </div>
    </div>

    <div class="metric-row">
      <div class="metric">
        <div class="label">Overall Outcome Score</div>
        <div class="value" id="v-total">&mdash;</div>
      </div>
    </div>
    <div class="callout callout-neutral">
      <div class="callout-label">How to read the score</div>
      A positive score means the strategy builds more sovereignty than it loses to dependency. A negative score means dependency costs outweigh the gains. Try dragging the first slider to see how different allocations change the outcome.
    </div>

    <div id="trajectory-chart" class="chart"></div>
    <div class="chart-caption">The chart updates in real time as you move the sliders. Watch how shifting resources toward future sovereignty (lower first slider) causes a short-term dip in present sovereignty but builds long-term capability &mdash; while dependency gradually falls.</div>

    <hr class="section-divider">

    <h2>The &ldquo;Sweet Spot&rdquo; of Designed Pressure</h2>
    <div class="prose">
      <p>The essay argues that <strong>not all constraint is equal</strong>. Broad sanctions tend to destroy innovation. But carefully designed pressure &mdash; like requiring R&amp;D teams to develop alternatives to foreign inputs &mdash; can actually <em>increase</em> the productivity of exploration. This is the &ldquo;induced innovation&rdquo; effect.</p>
      <p>The curve below shows this relationship. There is a sweet spot (marked by the dashed line at your current setting) where constraint intensity maximises innovation productivity. Too little pressure and nothing changes. Too much and the system breaks down.</p>
    </div>
    <div id="hump-chart" class="chart"></div>
    <div class="chart-caption">The peak of this curve is the optimal level of designed constraint. The dashed line shows your current setting. Notice that even very high constraint still produces <em>some</em> innovation &mdash; but far less than the optimal level.</div>
  </div>

  <!-- ============================================================ -->
  <!-- TAB 2: EVOLUTIONARY SIMULATION -->
  <!-- ============================================================ -->
  <div id="evolutionary" class="tab-content">
    <h2>What Happens to Individual Firms?</h2>
    <div class="prose">
      <p>The first tab treats the economy as a single decision-maker. But in reality, economies are made up of hundreds of firms with different capabilities, different levels of dependency on foreign technology, and different capacities to innovate.</p>
      <p>This simulation populates a virtual economy with firms and lets them evolve under different constraint regimes. Each firm can pursue <strong>exploitation</strong> (leveraging existing foreign technology for incremental gains) or <strong>exploration</strong> (developing novel alternatives). The constraint regime affects which strategy pays off.</p>
    </div>

    <div class="callout callout-insight">
      <div class="callout-label">What to look for</div>
      Compare the three default scenarios: <strong style="color:#b0a896;">No pressure</strong> (firms drift toward dependency), <strong style="color:#5a8a6a;">Designed pressure</strong> (capability dips then compounds as firms redirect), and <strong style="color:#c4756a;">Excessive pressure</strong> (most firms cannot adapt, capability degrades). The chart shows total innovation capacity of the economy over time.
    </div>

    <div class="controls">
      <div class="control">
        <label>Constraint levels to compare</label>
        <input type="text" id="evo-sigmas" value="0, 2, 15" style="width:160px; padding:0.3rem; font-size:0.8rem; border:1px solid #ccc; border-radius:3px;">
        <div class="slider-explain">0 = no constraint, 2 = designed, 15 = excessive. Enter any values separated by commas.</div>
      </div>
      <div class="control">
        <label>Number of firms in the economy</label>
        <input type="range" id="evo-N" min="50" max="500" step="50" value="200" oninput="document.getElementById('evo-N-val').textContent=this.value">
        <span class="val" id="evo-N-val">200</span>
        <div class="slider-explain">More firms = more realistic but slower to compute.</div>
      </div>
      <div class="control" style="justify-content:flex-end">
        <button class="run-btn" onclick="runEvolutionary()">Run Simulation</button>
      </div>
    </div>
    <div id="evo-chart" class="chart"></div>
    <div class="chart-caption">Each line represents the total innovation capacity of the economy under a different constraint regime. Under designed pressure (green), expect an initial dip &mdash; some firms struggle &mdash; followed by recovery as capable firms develop alternative pathways. This is the &ldquo;dip then compound&rdquo; pattern the essay describes.</div>

    <div class="callout callout-warning">
      <div class="callout-label">Key insight</div>
      Designed pressure only works when firms have sufficient &ldquo;absorptive capacity&rdquo; &mdash; enough existing capability to redirect rather than simply fail. This is why the essay insists that programme design must account for the capability distribution, not just set a constraint level and hope for the best.
    </div>
  </div>

  <!-- ============================================================ -->
  <!-- TAB 3: POLICY INTERFACE -->
  <!-- ============================================================ -->
  <div id="policy" class="tab-content">
    <h2>What Should Europe Actually Do?</h2>
    <div class="prose">
      <p>This view translates the model&rsquo;s results into policy-relevant comparisons. It answers three questions:</p>
      <p><strong>1. How do different strategies compare?</strong> The chart below shows 30-year trajectories under three regimes: the status quo (keep spending mostly on present sovereignty, no designed constraint), the model&rsquo;s recommended approach (rebalance toward exploration with designed constraint), and an overreaction scenario (excessive constraint, too little exploitation).</p>
    </div>

    <h3>Three Strategies, Three Futures</h3>
    <div class="read-guide">
      <div class="read-guide-item"><div class="guide-swatch" style="background:#b0a896;"></div> <strong>Status Quo</strong> &mdash; heavy exploitation, no constraint (presentism)</div>
      <div class="read-guide-item"><div class="guide-swatch" style="background:#5a8a6a;"></div> <strong>Designed Pressure</strong> &mdash; optimal allocation + strategic constraint</div>
      <div class="read-guide-item"><div class="guide-swatch" style="background:#c4756a;"></div> <strong>Overreaction</strong> &mdash; too much constraint, too little exploitation</div>
    </div>
    <div id="regime-chart" class="chart"></div>
    <div class="chart-caption">Solid lines = present sovereignty (control over today&rsquo;s tech). Dashed lines = future sovereignty (capacity in emerging domains). The designed pressure regime sacrifices some present sovereignty but builds far more future capability &mdash; and is the only strategy with a positive overall score.</div>
    <div class="regime-metrics" id="regime-metrics"></div>

    <hr class="section-divider">

    <h3>2. What Does Delay Cost?</h3>
    <div class="prose">
      <p>The essay argues that opportunity windows are closing. This chart quantifies the cost: if Europe waits 1 year, 5 years, or 10 years before shifting resources toward exploration, how much worse is the outcome? The curve accelerates because windows close exponentially &mdash; <strong>each year of inaction costs more than the last</strong>.</p>
    </div>
    <div id="delay-chart" class="chart"></div>
    <div class="chart-caption">The shaded area shows cumulative lost value from waiting. The steep acceleration after year 5 reflects paradigm windows closing: once standards lock in and incumbency advantages compound, the opportunity to build alternatives narrows sharply.</div>

    <hr class="section-divider">

    <h3>3. Where Should Europe Invest?</h3>
    <div class="prose">
      <p>Not all domains are equally promising. The model scores each by three factors: <strong>how open the opportunity window still is</strong>, <strong>how strong Europe&rsquo;s existing capability is</strong> in that area, and <strong>how strategically valuable</strong> the domain would be if Europe established a leading position. The overall score weights these 40/30/30.</p>
    </div>
    <div id="domain-bars"></div>

    <div class="callout callout-insight">
      <div class="callout-label">Reading the scores</div>
      <strong>Fusion energy</strong> scores highest because its window opened recently (more time remaining) and its strategic value is very high. <strong>Scientific AI</strong> and <strong>quantum computing</strong> are close, but with different profiles: quantum benefits from stronger existing European research capacity, while scientific AI has a wider remaining window. The essay argues that these are domains &ldquo;where the game is not yet rigged by incumbency.&rdquo;
    </div>
  </div>
</div>

<script>
var C = {drift:'#b0a896',destructive:'#c4756a',designed:'#5a8a6a',present:'#8B4049',future:'#5a7a8a',dependency:'#b89a3c'};
var plotLayout = {paper_bgcolor:'#FAFAF7',plot_bgcolor:'#FAFAF7',font:{family:'Inter,sans-serif',size:11},margin:{t:30,b:40,l:50,r:20},xaxis:{gridcolor:'#e8e6e1'},yaxis:{gridcolor:'#e8e6e1'}};

// Build tabs
(function() {
  var tabs = [
    {id:'analytical', label:'Layer 1: Analytical Core'},
    {id:'evolutionary', label:'Layer 2: Evolutionary Simulation'},
    {id:'policy', label:'Layer 3: Policy Interface'}
  ];
  var bar = document.getElementById('tab-bar');
  tabs.forEach(function(t, i) {
    var el = document.createElement('div');
    el.className = 'tab' + (i === 0 ? ' active' : '');
    el.textContent = t.label;
    el.setAttribute('data-target', t.id);
    el.addEventListener('click', function() { switchTab(t.id, el); });
    bar.appendChild(el);
  });
})();

function switchTab(id, tabEl) {
  document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
  document.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
  document.getElementById(id).classList.add('active');
  tabEl.classList.add('active');
  if (id === 'policy') { loadPolicy(); }
}

function getParams() {
  return 'D0=' + gv('D0') + '&R=0.022&T=30&gamma=' + gv('gamma') + '&rho=0.03';
}
function gv(id) { return document.getElementById(id).value; }

function updateAnalytical() {
  ['alpha','sigma','D0','gamma'].forEach(function(id) {
    document.getElementById(id + '-val').textContent = parseFloat(gv(id)).toFixed(2);
  });
  fetch('/api/simulate?alpha=' + gv('alpha') + '&sigma=' + gv('sigma') + '&' + getParams())
    .then(function(r) { return r.json(); })
    .then(function(r) {
      document.getElementById('v-total').textContent = r.V_total.toFixed(4);
      Plotly.react('trajectory-chart', [
        {x:r.t,y:r.Kp,name:'Kp',line:{color:C.present,width:2}},
        {x:r.t,y:r.Kf,name:'Kf',line:{color:C.future,width:2}},
        {x:r.t,y:r.D,name:'D',line:{color:C.dependency,width:2},yaxis:'y2'}
      ], Object.assign({}, plotLayout, {
        height:350,
        yaxis:Object.assign({}, plotLayout.yaxis, {title:'Kp, Kf'}),
        yaxis2:{overlaying:'y',side:'right',title:'D',range:[0,1],gridcolor:'transparent'},
        xaxis:Object.assign({}, plotLayout.xaxis, {title:'Years'}),
        legend:{x:0,y:1}
      }));
    });
  fetch('/api/hump').then(function(r) { return r.json(); }).then(function(h) {
    var sig = parseFloat(gv('sigma'));
    Plotly.react('hump-chart', [
      {x:h.sigmas,y:h.h_values,line:{color:C.designed,width:2},name:'h(sigma)'}
    ], Object.assign({}, plotLayout, {
      height:280,
      xaxis:Object.assign({}, plotLayout.xaxis, {title:'sigma'}),
      yaxis:Object.assign({}, plotLayout.yaxis, {title:'h(sigma)'}),
      shapes:[{type:'line',x0:sig,x1:sig,y0:0,y1:1,yref:'paper',line:{color:C.destructive,dash:'dash',width:1.5}}],
      annotations:[{x:sig,y:1,yref:'paper',text:'sigma='+sig,showarrow:false,font:{size:10,color:C.destructive}}]
    }));
  });
}

function runEvolutionary() {
  var sigmas = document.getElementById('evo-sigmas').value.split(',').map(function(s) { return parseFloat(s.trim()); });
  var N = parseInt(gv('evo-N'));
  var colorMap = {'0':'#b0a896', '2':'#5a8a6a', '15':'#c4756a'};
  var promises = sigmas.map(function(s) {
    return fetch('/api/evolutionary?sigma=' + s + '&N=' + N).then(function(r) { return r.json(); }).then(function(r) { return {sigma: s, data: r}; });
  });
  Promise.all(promises).then(function(results) {
    var traces = results.map(function(r) {
      return {x:r.data.t, y:r.data.aggregate_capability, name:'sigma = '+r.sigma, line:{color:colorMap[String(r.sigma)]||'#666',width:2}};
    });
    Plotly.react('evo-chart', traces, Object.assign({}, plotLayout, {
      height:400,
      xaxis:Object.assign({}, plotLayout.xaxis, {title:'Years'}),
      yaxis:Object.assign({}, plotLayout.yaxis, {title:'Aggregate Capability C(t)'})
    }));
  });
}

function loadPolicy() {
  var rc = {status_quo:'#b0a896',designed:'#5a8a6a',overreaction:'#c4756a'};
  var rl = {status_quo:'Status Quo',designed:'Designed Pressure',overreaction:'Overreaction'};
  fetch('/api/regimes?' + getParams()).then(function(r) { return r.json(); }).then(function(reg) {
    var traces = [];
    var metricsEl = document.getElementById('regime-metrics');
    while (metricsEl.firstChild) metricsEl.removeChild(metricsEl.firstChild);
    Object.keys(reg).forEach(function(k) {
      var r = reg[k];
      traces.push({x:r.t,y:r.Kp,name:rl[k]+' Kp',line:{color:rc[k],width:2}});
      traces.push({x:r.t,y:r.Kf,name:rl[k]+' Kf',line:{color:rc[k],width:2,dash:'dash'}});
      var bg = k==='designed'?'#e5f0e8':k==='overreaction'?'#fce8e5':'#f0ece4';
      var descs = {status_quo:'Exploit-heavy, no constraint',designed:'Balanced allocation + strategic pressure',overreaction:'Excessive constraint, too little exploitation'};
      var div = document.createElement('div');
      div.className = 'regime-metric';
      div.style.background = bg;
      var lbl = document.createElement('div');
      lbl.className = 'rlabel';
      lbl.textContent = rl[k];
      var desc = document.createElement('div');
      desc.className = 'rdesc';
      desc.textContent = descs[k];
      var val = document.createElement('div');
      val.className = 'rvalue';
      val.textContent = 'Score: ' + r.V_total.toFixed(2);
      div.appendChild(lbl);
      div.appendChild(desc);
      div.appendChild(val);
      metricsEl.appendChild(div);
    });
    Plotly.react('regime-chart', traces, Object.assign({}, plotLayout, {
      height:350,
      xaxis:Object.assign({}, plotLayout.xaxis, {title:'Years'}),
      yaxis:Object.assign({}, plotLayout.yaxis, {title:'Sovereignty Capital'}),
      legend:{font:{size:9}}
    }));
  });
  fetch('/api/delay?' + getParams()).then(function(r) { return r.json(); }).then(function(d) {
    Plotly.react('delay-chart', [{x:d.delays,y:d.costs,fill:'tozeroy',fillcolor:'rgba(139,64,73,0.15)',line:{color:C.present,width:2}}], Object.assign({}, plotLayout, {
      height:280,
      xaxis:Object.assign({}, plotLayout.xaxis, {title:'Delay (years)'}),
      yaxis:Object.assign({}, plotLayout.yaxis, {title:'Cumulative cost'})
    }));
  });
  fetch('/api/domains').then(function(r) { return r.json(); }).then(function(domains) {
    var container = document.getElementById('domain-bars');
    while (container.firstChild) container.removeChild(container.firstChild);
    domains.forEach(function(d) {
      var pct = (d.score * 100).toFixed(0);
      var wrapper = document.createElement('div');
      wrapper.className = 'domain-bar';
      var lbl = document.createElement('div');
      lbl.className = 'bar-label';
      var strong = document.createElement('strong');
      strong.textContent = d.name;
      lbl.appendChild(strong);
      var track = document.createElement('div');
      track.className = 'bar-track';
      var fill = document.createElement('div');
      fill.className = 'bar-fill';
      fill.style.width = pct + '%';
      var span = document.createElement('span');
      span.textContent = d.score.toFixed(2);
      fill.appendChild(span);
      track.appendChild(fill);
      var detail = document.createElement('div');
      detail.className = 'bar-detail';
      detail.textContent = 'Window remaining: ' + (d.W_remaining * 100).toFixed(0) + '% \u00b7 Existing capability: ' + (d.capability * 100).toFixed(0) + '% \u00b7 Strategic value: ' + (d.strategic_value * 100).toFixed(0) + '%';
      wrapper.appendChild(lbl);
      wrapper.appendChild(track);
      wrapper.appendChild(detail);
      container.appendChild(wrapper);
    });
  });
}

updateAnalytical();
</script>
</body>
</html>"""
