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
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Source Serif 4', Georgia, serif; background: #FAFAF7; color: #1a1a1a; }
  .page { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { font-family: 'Playfair Display', serif; font-size: 1.8rem; margin-bottom: 0.25rem; }
  h2 { font-family: 'Playfair Display', serif; font-size: 1.3rem; margin: 2rem 0 0.5rem; }
  .subtitle { font-size: 0.95rem; color: #777; font-style: italic; margin-bottom: 1rem; }
  .pdf-link { display: inline-block; font-family: 'Inter', sans-serif; font-size: 0.8rem; background: #5a8a6a; color: white; padding: 0.4rem 1rem; border-radius: 4px; text-decoration: none; margin: 0.75rem 0; }
  .pdf-link:hover { background: #4a7a5a; }
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
  .domain-bar { margin: 0.5rem 0; }
  .domain-bar .bar-label { font-family: 'Inter', sans-serif; font-size: 0.8rem; margin-bottom: 0.2rem; }
  .domain-bar .bar-track { height: 24px; background: #f0ece4; border-radius: 3px; overflow: hidden; }
  .domain-bar .bar-fill { height: 100%; border-radius: 3px; background: #5a8a6a; display: flex; align-items: center; padding-left: 0.5rem; }
  .domain-bar .bar-fill span { font-family: 'Inter', sans-serif; font-size: 0.7rem; color: white; font-weight: 600; }
  .regime-metrics { display: flex; gap: 1rem; margin: 1rem 0; }
  .regime-metric { flex: 1; text-align: center; padding: 0.75rem; border-radius: 4px; }
  .regime-metric .rlabel { font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #666; }
  .regime-metric .rvalue { font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 600; margin-top: 0.2rem; }
  button.run-btn { font-family: 'Inter', sans-serif; font-size: 0.8rem; padding: 0.5rem 1rem; background: #5a8a6a; color: white; border: none; border-radius: 4px; cursor: pointer; }
  button.run-btn:hover { background: #4a7a5a; }
</style>
</head>
<body>
<div class="page">
  <h1>Two Kinds of Sovereignty &mdash; Economic Model</h1>
  <p class="subtitle">Interactive companion to the formal model. Explore how allocation, constraint intensity, and window dynamics shape sovereignty trajectories.</p>
  <a href="/pdf" class="pdf-link">Download formal model specification (PDF)</a>

  <div class="tabs" id="tab-bar"></div>

  <div id="analytical" class="tab-content active">
    <h2>Optimal Allocation &amp; Constraint Design</h2>
    <div class="controls">
      <div class="control">
        <label>Exploitation share &alpha;</label>
        <input type="range" id="alpha" min="0" max="1" step="0.01" value="0.5" oninput="updateAnalytical()">
        <span class="val" id="alpha-val">0.50</span>
      </div>
      <div class="control">
        <label>Constraint intensity &sigma;</label>
        <input type="range" id="sigma" min="0" max="15" step="0.1" value="2.0" oninput="updateAnalytical()">
        <span class="val" id="sigma-val">2.0</span>
      </div>
      <div class="control">
        <label>Initial dependency D&#8320;</label>
        <input type="range" id="D0" min="0" max="1" step="0.01" value="0.67" oninput="updateAnalytical()">
        <span class="val" id="D0-val">0.67</span>
      </div>
      <div class="control">
        <label>Window closure rate &gamma;</label>
        <input type="range" id="gamma" min="0.01" max="0.3" step="0.01" value="0.1" oninput="updateAnalytical()">
        <span class="val" id="gamma-val">0.10</span>
      </div>
    </div>
    <div class="metric-row">
      <div class="metric"><div class="label">Discounted Total Value</div><div class="value" id="v-total">&mdash;</div></div>
    </div>
    <div id="trajectory-chart" class="chart"></div>
    <h2>Constraint Instrument h(&sigma;)</h2>
    <div id="hump-chart" class="chart"></div>
  </div>

  <div id="evolutionary" class="tab-content">
    <h2>Selection Pressure Dynamics</h2>
    <div class="controls">
      <div class="control">
        <label>&sigma; values (comma-separated)</label>
        <input type="text" id="evo-sigmas" value="0, 2, 15" style="width:160px; padding:0.3rem; font-size:0.8rem; border:1px solid #ccc; border-radius:3px;">
      </div>
      <div class="control">
        <label>Number of firms</label>
        <input type="range" id="evo-N" min="50" max="500" step="50" value="200" oninput="document.getElementById('evo-N-val').textContent=this.value">
        <span class="val" id="evo-N-val">200</span>
      </div>
      <div class="control" style="justify-content:flex-end">
        <button class="run-btn" onclick="runEvolutionary()">Run Simulation</button>
      </div>
    </div>
    <div id="evo-chart" class="chart"></div>
  </div>

  <div id="policy" class="tab-content">
    <h2>Three-Regime Comparison</h2>
    <div id="regime-chart" class="chart"></div>
    <div class="regime-metrics" id="regime-metrics"></div>
    <h2>Shadow Price of Delay</h2>
    <div id="delay-chart" class="chart"></div>
    <h2>Domain Priority Scoring</h2>
    <div id="domain-bars"></div>
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
      var div = document.createElement('div');
      div.className = 'regime-metric';
      div.style.background = bg;
      var lbl = document.createElement('div');
      lbl.className = 'rlabel';
      lbl.textContent = rl[k];
      var val = document.createElement('div');
      val.className = 'rvalue';
      val.textContent = 'V = ' + r.V_total.toFixed(4);
      div.appendChild(lbl);
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
      lbl.appendChild(document.createTextNode(' \u2014 Window: ' + d.W_remaining + ', Capability: ' + d.capability + ', Strategic: ' + d.strategic_value));
      var track = document.createElement('div');
      track.className = 'bar-track';
      var fill = document.createElement('div');
      fill.className = 'bar-fill';
      fill.style.width = pct + '%';
      var span = document.createElement('span');
      span.textContent = d.score.toFixed(2);
      fill.appendChild(span);
      track.appendChild(fill);
      wrapper.appendChild(lbl);
      wrapper.appendChild(track);
      container.appendChild(wrapper);
    });
  });
}

updateAnalytical();
</script>
</body>
</html>"""
