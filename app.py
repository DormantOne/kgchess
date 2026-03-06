#!/usr/bin/env python3
"""
Escape-Check KG — Web GUI
==========================
Flask app with three modes:
  1. TRAIN   — configure params (incl. Ollama model auto-detect), run training
  2. TEST    — query KG with FEN, click suggestions to see moves on the board
  3. CONTROL — batch comparison: trained vs untrained KG vs heuristic baseline

Run:
  python app.py [--port 5000] [--db kg.sqlite]
"""

import json
import os
import random
import threading
import time
import sys

from flask import Flask, render_template_string, request, jsonify

# Add current dir to path so we can import engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (
    KG, OllamaClient, train, query_position, run_control_test,
    generate_random_check_position, heuristic_escapes,
    render_board_with_move,
)

import chess
import chess.svg

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get("KG_DB", "kg.sqlite")
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TRAIN_STATE = {
    "running": False,
    "progress": 0,
    "total": 0,
    "log": [],
    "summary": None,
}
CONTROL_STATE = {
    "running": False,
    "progress": 0,
    "total": 0,
    "log": [],
    "results": None,
}

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Escape-Check KG</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

  :root {
    --bg-deep: #0a0c10;
    --bg-card: #12151c;
    --bg-raised: #1a1e28;
    --bg-input: #0e1018;
    --border: #252a36;
    --border-bright: #3a4155;
    --text: #c8cdd8;
    --text-dim: #6b7280;
    --text-bright: #e8ecf4;
    --accent: #60a5fa;
    --accent-glow: rgba(96,165,250,0.15);
    --green: #34d399;
    --green-dim: rgba(52,211,153,0.12);
    --red: #f87171;
    --red-dim: rgba(248,113,113,0.12);
    --amber: #fbbf24;
    --amber-dim: rgba(251,191,36,0.12);
    --purple: #a78bfa;
    --radius: 10px;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'DM Sans', system-ui, sans-serif;
    background: var(--bg-deep);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
  }

  .header {
    background: linear-gradient(180deg, #111420 0%, var(--bg-deep) 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex; align-items: center; gap: 20px;
  }
  .header-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--accent), var(--purple));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: #fff;
  }
  .header h1 { font-size: 1.3rem; font-weight: 700; color: var(--text-bright); letter-spacing: -0.02em; }
  .header .subtitle { font-size: 0.82rem; color: var(--text-dim); font-weight: 400; }
  .header-stats {
    margin-left: auto;
    display: flex; gap: 24px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
  }
  .stat-pill {
    background: var(--bg-raised); border: 1px solid var(--border);
    border-radius: 20px; padding: 6px 14px;
    display: flex; align-items: center; gap: 6px;
  }
  .stat-pill .num { color: var(--accent); font-weight: 600; }

  .tab-bar { display: flex; background: var(--bg-card); border-bottom: 1px solid var(--border); padding: 0 32px; }
  .tab-btn {
    padding: 14px 24px; cursor: pointer; font-size: 0.88rem; font-weight: 500;
    color: var(--text-dim); border: none; background: none;
    border-bottom: 2px solid transparent; transition: all 0.2s; font-family: inherit;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; padding: 28px 32px; max-width: 1200px; }
  .tab-content.active { display: block; }

  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; margin-bottom: 20px; }
  .card-title { font-size: 0.92rem; font-weight: 600; color: var(--text-bright); margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  .card-title .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }

  .form-row { display: flex; gap: 16px; margin-bottom: 14px; flex-wrap: wrap; align-items: end; }
  .form-group { display: flex; flex-direction: column; gap: 5px; }
  .form-group label { font-size: 0.78rem; color: var(--text-dim); font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
  input[type="text"], input[type="number"], select {
    background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px;
    padding: 9px 12px; color: var(--text-bright);
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
    outline: none; transition: border-color 0.2s;
  }
  input:focus, select:focus { border-color: var(--accent); }
  input[type="number"] { width: 100px; }

  .btn {
    padding: 10px 22px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--bg-raised); color: var(--text-bright); font-weight: 600;
    font-size: 0.85rem; cursor: pointer; transition: all 0.15s; font-family: inherit;
  }
  .btn:hover { background: var(--border); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #0a0c10; border-color: var(--accent); }
  .btn-primary:hover { background: #7bb8fc; }
  .btn-amber { background: var(--amber); color: #0a0c10; border-color: var(--amber); }
  .btn-amber:hover { background: #fcd34d; }
  .btn-sm { padding: 6px 14px; font-size: 0.78rem; }

  .progress-bar-track { width: 100%; height: 6px; background: var(--bg-input); border-radius: 3px; overflow: hidden; margin: 12px 0; }
  .progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--purple)); border-radius: 3px; transition: width 0.3s; width: 0%; }
  .progress-text { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-dim); }

  .console {
    background: var(--bg-input); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;
    line-height: 1.7; max-height: 260px; overflow-y: auto; color: var(--text-dim);
  }
  .console .line-ok { color: var(--green); }
  .console .line-warn { color: var(--amber); }
  .console .line-info { color: var(--accent); }

  .board-area { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-start; }
  .board-svg { flex-shrink: 0; }
  .board-svg svg { border-radius: 8px; }
  .suggestions-panel { flex: 1; min-width: 300px; }

  .sug-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; background: var(--bg-raised); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    cursor: pointer; transition: all 0.15s; user-select: none;
  }
  .sug-row:hover { border-color: var(--accent); background: rgba(96,165,250,0.06); }
  .sug-row.active { border-color: var(--accent); background: rgba(96,165,250,0.10); box-shadow: 0 0 12px rgba(96,165,250,0.1); }
  .sug-rank {
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--accent-glow); color: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.78rem;
  }
  .sug-move { font-weight: 700; color: var(--text-bright); width: 60px; }
  .sug-bar { flex: 1; height: 6px; background: var(--bg-input); border-radius: 3px; overflow: hidden; }
  .sug-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .sug-score { width: 50px; text-align: right; color: var(--text-dim); }

  .board-pair { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; }
  .board-col { text-align: center; }
  .board-col svg { border-radius: 8px; }
  .board-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-dim); margin-bottom: 6px;
  }
  .board-label .move-tag {
    display: inline-block; background: var(--accent-glow); color: var(--accent);
    padding: 2px 8px; border-radius: 8px; font-family: 'JetBrains Mono', monospace;
    margin-left: 6px;
  }
  .move-info-badges { display: flex; gap: 6px; justify-content: center; margin-top: 8px; }

  .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 20px; }
  .result-box { background: var(--bg-raised); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px; text-align: center; }
  .result-box .label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
  .result-box .value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; }
  .result-box .value.green { color: var(--green); }
  .result-box .value.amber { color: var(--amber); }
  .result-box .value.red { color: var(--red); }
  .result-box .value.blue { color: var(--accent); }

  .features-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
  .feat-item { background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 0.78rem; }
  .feat-item .feat-key { color: var(--text-dim); }
  .feat-item .feat-val { color: var(--text-bright); font-family: 'JetBrains Mono', monospace; font-weight: 600; }

  .empty-state { text-align: center; padding: 40px; color: var(--text-dim); font-size: 0.9rem; }
  .empty-state .icon { font-size: 2rem; margin-bottom: 10px; }
  .fen-display {
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 14px; color: var(--accent); word-break: break-all; margin: 10px 0;
  }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .badge-green { background: var(--green-dim); color: var(--green); }
  .badge-red { background: var(--red-dim); color: var(--red); }
  .badge-amber { background: var(--amber-dim); color: var(--amber); }
  .badge-blue { background: var(--accent-glow); color: var(--accent); }

  .ollama-status {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.78rem; padding: 8px 14px;
    background: var(--bg-input); border: 1px solid var(--border);
    border-radius: 8px; margin-top: 10px;
  }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .status-dot.on { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .status-dot.off { background: var(--red); box-shadow: 0 0 6px var(--red); }
  .status-dot.checking { background: var(--amber); animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
</head>
<body>

<div class="header">
  <div class="header-icon">&#9818;</div>
  <div>
    <h1>Escape-Check KG</h1>
    <div class="subtitle">Knowledge graph trainer for check evasion patterns</div>
  </div>
  <div class="header-stats">
    <div class="stat-pill">Nodes <span class="num" id="stat-nodes">&mdash;</span></div>
    <div class="stat-pill">Edges <span class="num" id="stat-edges">&mdash;</span></div>
    <div class="stat-pill">Episodes <span class="num" id="stat-episodes">&mdash;</span></div>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('train', this)">Train</button>
  <button class="tab-btn" onclick="switchTab('test', this)">Test</button>
  <button class="tab-btn" onclick="switchTab('control', this)">Control Test</button>
</div>

<!-- ==================== TRAIN TAB ==================== -->
<div class="tab-content active" id="tab-train">
  <div class="card">
    <div class="card-title"><span class="dot"></span>Training Configuration</div>
    <div class="form-row">
      <div class="form-group"><label>Episodes</label><input type="number" id="train-episodes" value="200" min="10" max="5000"></div>
      <div class="form-group"><label>Seed</label><input type="number" id="train-seed" value="1" min="0"></div>
      <div class="form-group"><label>Max Plies</label><input type="number" id="train-plies" value="60" min="10" max="200"></div>
      <div class="form-group"><label>Oracle Depth</label><input type="number" id="train-depth" value="10" min="1" max="30"></div>
    </div>
    <div class="card-title" style="margin-top:8px"><span class="dot" style="background:var(--purple)"></span>Ollama LLM Proposer</div>
    <div class="form-row">
      <div class="form-group"><label>Enable</label>
        <select id="train-ollama" onchange="onOllamaToggle()"><option value="off">Off</option><option value="on">On</option></select>
      </div>
      <div class="form-group" id="ollama-host-group" style="display:none"><label>Host</label>
        <input type="text" id="train-ollama-host" value="http://localhost:11434" style="width:220px">
      </div>
      <div class="form-group" id="ollama-model-group" style="display:none"><label>Model</label>
        <select id="train-ollama-model" style="min-width:200px"><option value="gpt-oss:20b">gpt-oss:20b (default)</option></select>
      </div>
      <div class="form-group" id="ollama-check-group" style="display:none;justify-content:end">
        <button class="btn btn-sm" onclick="checkOllama()">Check Connection</button>
        <button class="btn btn-sm" onclick="debugOllama()" style="margin-left:6px;background:var(--purple);color:#fff;border-color:var(--purple)">Debug Test</button>
      </div>
    </div>
    <div id="ollama-status-area" style="display:none"></div>
    <div class="form-row" style="margin-top:16px">
      <button class="btn btn-primary" id="btn-train" onclick="startTraining()">Start Training</button>
    </div>
  </div>
  <div class="card" id="train-progress-card" style="display:none">
    <div class="card-title"><span class="dot" style="background:var(--green)"></span>Training Progress</div>
    <div style="display:flex;align-items:center;gap:16px;">
      <span class="progress-text" id="train-pct">0%</span>
      <div class="progress-bar-track" style="flex:1"><div class="progress-bar-fill" id="train-bar"></div></div>
      <span class="progress-text" id="train-count">0 / 0</span>
    </div>
    <div class="console" id="train-log"></div>
  </div>
  <div class="card" id="train-summary-card" style="display:none">
    <div class="card-title"><span class="dot" style="background:var(--green)"></span>Training Summary</div>
    <div class="results-grid" id="train-summary"></div>
  </div>
</div>

<!-- ==================== TEST TAB ==================== -->
<div class="tab-content" id="tab-test">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--purple)"></span>Query Position</div>
    <div class="form-row">
      <div class="form-group" style="flex:1"><label>FEN (must be in check)</label>
        <input type="text" id="test-fen" style="width:100%" placeholder="Enter FEN or click Generate Random...">
      </div>
      <div class="form-group" style="justify-content:end"><button class="btn" onclick="generateRandom()">Generate Random</button></div>
      <div class="form-group" style="justify-content:end"><button class="btn btn-primary" onclick="queryKG()">Query KG</button></div>
    </div>
  </div>
  <div id="test-results"></div>
</div>

<!-- ==================== CONTROL TAB ==================== -->
<div class="tab-content" id="tab-control">
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--amber)"></span>Control Test: Trained vs Untrained KG</div>
    <p style="font-size:0.85rem;color:var(--text-dim);margin-bottom:16px;">
      Generates random check positions and compares the trained KG against Stockfish.
      An untrained (empty) KG serves as the control baseline.
    </p>
    <div class="form-row">
      <div class="form-group"><label>Positions</label><input type="number" id="ctrl-positions" value="50" min="10" max="500"></div>
      <div class="form-group"><label>Seed</label><input type="number" id="ctrl-seed" value="999" min="0"></div>
      <div class="form-group"><label>Oracle Depth</label><input type="number" id="ctrl-depth" value="10" min="1" max="30"></div>
      <div class="form-group" style="justify-content:end"><button class="btn btn-amber" id="btn-control" onclick="startControl()">Run Control Test</button></div>
    </div>
  </div>
  <div class="card" id="ctrl-progress-card" style="display:none">
    <div class="card-title"><span class="dot" style="background:var(--amber)"></span>Progress</div>
    <div style="display:flex;align-items:center;gap:16px;">
      <span class="progress-text" id="ctrl-pct">0%</span>
      <div class="progress-bar-track" style="flex:1"><div class="progress-bar-fill" id="ctrl-bar" style="background:linear-gradient(90deg,var(--amber),var(--green))"></div></div>
      <span class="progress-text" id="ctrl-count">0 / 0</span>
    </div>
    <div class="console" id="ctrl-log"></div>
  </div>
  <div id="ctrl-results"></div>
</div>

<script>
let currentTestFen = null;

function switchTab(name, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
  refreshStats();
}

async function refreshStats() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('stat-nodes').textContent = d.total_nodes;
    document.getElementById('stat-edges').textContent = d.total_edges;
    document.getElementById('stat-episodes').textContent = d.total_episodes;
  } catch(e) {}
}
refreshStats();

// ==================== OLLAMA ====================

function onOllamaToggle() {
  const on = document.getElementById('train-ollama').value === 'on';
  ['ollama-host-group','ollama-model-group','ollama-check-group'].forEach(id => {
    document.getElementById(id).style.display = on ? 'flex' : 'none';
  });
  document.getElementById('ollama-status-area').style.display = on ? 'block' : 'none';
  if (on) checkOllama();
}

async function checkOllama() {
  const host = document.getElementById('train-ollama-host').value.trim();
  const sa = document.getElementById('ollama-status-area');
  sa.innerHTML = '<div class="ollama-status"><span class="status-dot checking"></span><span style="color:var(--amber)">Checking ' + host + '...</span></div>';
  try {
    const r = await fetch('/api/ollama/models', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({host})});
    const d = await r.json();
    if (d.available && d.models.length > 0) {
      const sel = document.getElementById('train-ollama-model');
      const prev = sel.value;
      sel.innerHTML = '';
      d.models.forEach(m => {
        const o = document.createElement('option');
        o.value = m.name;
        o.textContent = m.name + (m.parameter_size ? ' (' + m.parameter_size + ')' : m.size ? ' (' + m.size + ')' : '');
        sel.appendChild(o);
      });
      const names = d.models.map(m => m.name);
      if (names.includes(prev)) sel.value = prev;
      else if (names.includes('gpt-oss:20b')) sel.value = 'gpt-oss:20b';
      sa.innerHTML = '<div class="ollama-status"><span class="status-dot on"></span><span style="color:var(--green)">Connected &mdash; ' + d.models.length + ' model(s) available</span></div>';
    } else if (d.available) {
      sa.innerHTML = '<div class="ollama-status"><span class="status-dot on"></span><span style="color:var(--amber)">Server up but no models. Run: ollama pull gpt-oss:20b</span></div>';
    } else {
      sa.innerHTML = '<div class="ollama-status"><span class="status-dot off"></span><span style="color:var(--red)">' + (d.error || 'Cannot connect') + '</span></div>';
    }
  } catch(e) {
    sa.innerHTML = '<div class="ollama-status"><span class="status-dot off"></span><span style="color:var(--red)">Check failed: ' + e + '</span></div>';
  }
}

async function debugOllama() {
  const host = document.getElementById('train-ollama-host').value.trim();
  const model = document.getElementById('train-ollama-model').value;
  const sa = document.getElementById('ollama-status-area');
  sa.innerHTML = '<div class="ollama-status"><span class="status-dot checking"></span><span style="color:var(--amber)">Running debug (ping + chess test)... may take 1-3 min</span></div>';
  try {
    const r = await fetch('/api/ollama/debug', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({host, model})});
    const d = await r.json();
    const S = (x) => typeof x === 'string' ? x : JSON.stringify(x, null, 2);
    let html = '<div style="background:var(--bg-input);border:1px solid var(--border);border-radius:8px;padding:16px;margin-top:12px;font-family:JetBrains Mono,monospace;font-size:0.72rem;line-height:1.5;max-height:600px;overflow-y:auto">';

    // PING TEST
    const p = d.ping || {};
    const pOk = p.response && p.response.length > 0;
    html += '<div style="color:'+(pOk?'var(--green)':'var(--red)')+';font-weight:700;font-size:0.85rem">TEST 1: SIMPLE PING '+(pOk?'✓ PASSED':'✗ FAILED')+'</div>';
    html += '<div style="color:var(--text-dim);margin:4px 0">Prompt: "'+escHtml(p.prompt||'')+'"</div>';
    html += '<div style="color:var(--green);margin:4px 0">Response: <b>'+escHtml(p.response||'(empty)')+'</b></div>';
    if (p.error) html += '<div style="color:var(--red)">Error: '+escHtml(p.error)+'</div>';
    html += '<details style="margin:4px 0"><summary style="color:var(--text-dim);cursor:pointer">Full API response</summary><pre style="white-space:pre-wrap;color:var(--text);background:var(--bg-deep);padding:8px;border-radius:4px;max-height:200px;overflow-y:auto">'+escHtml(S(p.full_api_response))+'</pre></details>';

    html += '<hr style="border-color:var(--border);margin:16px 0">';

    // CHESS TEST
    const c = d.chess || {};
    const cOk = c.response && c.response.length > 0;
    html += '<div style="color:'+(cOk?'var(--green)':'var(--red)')+';font-weight:700;font-size:0.85rem">TEST 2: CHESS PROMPT '+(cOk?'✓ GOT RESPONSE':'✗ EMPTY RESPONSE')+'</div>';
    html += '<div style="color:var(--text-dim);margin:4px 0">FEN: '+escHtml(c.fen||'?')+'</div>';
    html += '<div style="color:var(--text-dim);margin:4px 0">Legal moves: '+escHtml(JSON.stringify(c.legal_moves||[]))+'</div>';
    html += '<details style="margin:4px 0"><summary style="color:var(--purple);cursor:pointer;font-weight:600">Prompt sent</summary><pre style="white-space:pre-wrap;color:var(--text);background:var(--bg-deep);padding:8px;border-radius:4px">'+escHtml(c.prompt||'')+'</pre></details>';
    html += '<div style="color:var(--amber);font-weight:700;margin:8px 0">Raw LLM Response:</div>';
    html += '<pre style="white-space:pre-wrap;color:var(--green);background:var(--bg-deep);padding:10px;border-radius:6px;min-height:40px">'+escHtml(c.response||'(empty)')+'</pre>';
    if (c.error) html += '<div style="color:var(--red);margin:4px 0">Error: '+escHtml(c.error)+'</div>';
    html += '<details style="margin:4px 0"><summary style="color:var(--text-dim);cursor:pointer">Full API response</summary><pre style="white-space:pre-wrap;color:var(--text);background:var(--bg-deep);padding:8px;border-radius:4px;max-height:200px;overflow-y:auto">'+escHtml(S(c.full_api_response))+'</pre></details>';

    html += '</div>';
    sa.innerHTML = html;
  } catch(e) {
    sa.innerHTML = '<div class="ollama-status"><span class="status-dot off"></span><span style="color:var(--red)">Debug failed: ' + e + '</span></div>';
  }
}
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ==================== TRAINING ====================

async function startTraining() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true; btn.textContent = 'Training...';
  document.getElementById('train-progress-card').style.display = 'block';
  document.getElementById('train-summary-card').style.display = 'none';
  document.getElementById('train-log').innerHTML = '';
  const useOllama = document.getElementById('train-ollama').value === 'on';
  const params = {
    episodes: parseInt(document.getElementById('train-episodes').value),
    seed: parseInt(document.getElementById('train-seed').value),
    max_plies: parseInt(document.getElementById('train-plies').value),
    oracle_depth: parseInt(document.getElementById('train-depth').value),
    use_ollama: useOllama,
    ollama_model: document.getElementById('train-ollama-model').value,
    ollama_host: document.getElementById('train-ollama-host').value,
  };
  try {
    const r = await fetch('/api/train', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params)});
    const d = await r.json();
    if (d.error) { alert(d.error); btn.disabled=false; btn.textContent='Start Training'; return; }
  } catch(e) { alert(e); btn.disabled=false; btn.textContent='Start Training'; return; }
  const poll = setInterval(async () => {
    try {
      const r = await fetch('/api/train/status'); const d = await r.json();
      const pct = d.total > 0 ? Math.round(100*d.progress/d.total) : 0;
      document.getElementById('train-pct').textContent = pct + '%';
      document.getElementById('train-bar').style.width = pct + '%';
      document.getElementById('train-count').textContent = d.progress + ' / ' + d.total;
      const logEl = document.getElementById('train-log');
      (d.log||[]).forEach(line => {
        const div = document.createElement('div');
        div.className = line.includes('match') ? 'line-ok' : line.includes('skip') ? 'line-warn' : 'line-info';
        div.textContent = line; logEl.appendChild(div);
      });
      if (d.log && d.log.length) logEl.scrollTop = logEl.scrollHeight;
      if (!d.running) { clearInterval(poll); btn.disabled=false; btn.textContent='Start Training'; refreshStats(); if(d.summary) showTrainSummary(d.summary); }
    } catch(e) {}
  }, 500);
}

function showTrainSummary(s) {
  document.getElementById('train-summary-card').style.display = 'block';
  const mr = s.episodes_completed > 0 ? Math.round(100*s.best_move_matches/s.episodes_completed) : 0;
  let oh = '';
  if (s.ollama_active) {
    oh = '<div class="result-box"><div class="label">LLM Proposals</div><div class="value blue">'+(s.ollama_proposals||0)+'</div></div>'
       + '<div class="result-box"><div class="label">LLM Unique Finds</div><div class="value '+((s.ollama_unique_finds||0)>0?'green':'amber')+'">'+(s.ollama_unique_finds||0)+'</div></div>';
  }
  document.getElementById('train-summary').innerHTML =
    '<div class="result-box"><div class="label">Completed</div><div class="value blue">'+s.episodes_completed+'</div></div>'
    +'<div class="result-box"><div class="label">Skipped</div><div class="value '+(s.episodes_skipped>s.episodes_completed/2?'amber':'green')+'">'+s.episodes_skipped+'</div></div>'
    +'<div class="result-box"><div class="label">Situations</div><div class="value blue">'+s.unique_situations+'</div></div>'
    +'<div class="result-box"><div class="label">Oracle Match</div><div class="value '+(mr>30?'green':'amber')+'">'+mr+'%</div></div>'
    +'<div class="result-box"><div class="label">Oracle</div><div class="value '+(s.oracle_active?'green':'red')+'">'+(s.oracle_active?'ON':'OFF')+'</div></div>'
    +'<div class="result-box"><div class="label">Ollama</div><div class="value '+(s.ollama_active?'green':'amber')+'">'+(s.ollama_active?'ON':'OFF')+'</div></div>'
    +oh;
}

// ==================== TEST ====================

async function generateRandom() {
  try {
    const r = await fetch('/api/random-check'); const d = await r.json();
    document.getElementById('test-fen').value = d.fen || '';
    if (d.fen) queryKG();
  } catch(e) { alert(e); }
}

async function queryKG() {
  const fen = document.getElementById('test-fen').value.trim();
  if (!fen) { alert('Enter a FEN'); return; }
  try {
    const r = await fetch('/api/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({fen})});
    const d = await r.json();
    if (d.error) { alert(d.error); return; }
    currentTestFen = fen;
    renderTestResult(d);
  } catch(e) { alert(e); }
}

function renderTestResult(d) {
  const c = document.getElementById('test-results');
  if (!d.in_check) { c.innerHTML = '<div class="card"><div class="empty-state"><div class="icon">&#9888;</div>Not in check.</div></div>'; return; }
  const f = d.features || {};
  const fi = [['Check Type',f.check_type],['Side',f.side_to_move==='w'?'White':'Black'],['Checkers',f.num_checkers],['King Moves',f.safe_king_moves],['Captures',f.capture_options],['Blocks',f.block_options],['Double',f.double_check?'Yes':'No'],['Material',f.material_balance_rough]]
    .map(([k,v])=>'<div class="feat-item"><span class="feat-key">'+k+'</span><br><span class="feat-val">'+v+'</span></div>').join('');
  let sh = '';
  if (d.suggestions && d.suggestions.length) {
    sh = d.suggestions.map((s,i)=>{
      const sc = Math.round((s.robustness*0.5+s.oracle_signal*0.3+s.gives_check*0.2)*100);
      const co = sc>60?'var(--green)':sc>30?'var(--amber)':'var(--red)';
      return '<div class="sug-row" onclick="previewMove(\''+s.move+'\',this)">'
        +'<div class="sug-rank">'+(i+1)+'</div><div class="sug-move">'+s.move+'</div>'
        +'<div style="display:flex;gap:6px;flex-wrap:wrap;flex:1;font-size:0.72rem">'
        +'<span class="badge badge-green">robust '+(s.robustness*100).toFixed(0)+'%</span>'
        +'<span class="badge badge-blue">oracle '+(s.oracle_signal*100).toFixed(0)+'%</span>'
        +(s.gives_check>0?'<span class="badge badge-amber">gives check!</span>':'')
        +'</div><div class="sug-bar"><div class="sug-bar-fill" style="width:'+sc+'%;background:'+co+'"></div></div>'
        +'<div class="sug-score" style="color:'+co+'">'+s.edge_hp+'</div></div>';
    }).join('');
  } else {
    sh = '<div class="empty-state" style="padding:24px"><div class="icon">&#128269;</div>No learned suggestions. Train more or try another position.</div>';
  }
  const ni = d.node_found
    ? '<span class="badge badge-green">Node Found</span> <span style="font-family:JetBrains Mono;font-size:0.8rem;color:var(--text-dim)">HP: '+d.node_hp+'</span>'
      + (d.illegal_filtered > 0 ? ' <span class="badge badge-amber">'+d.illegal_filtered+' illegal filtered</span>' : '')
    : '<span class="badge badge-red">No Node</span>';
  let hh = '';
  if (d.heuristic_moves && d.heuristic_moves.length) {
    hh = '<div style="margin-top:14px;font-size:0.8rem;color:var(--text-dim)"><strong>Heuristic:</strong> '
      + d.heuristic_moves.map(m=>'<span style="cursor:pointer;color:var(--accent);text-decoration:underline" onclick="previewMove(\''+m+'\',null)">'+m+'</span>').join(', ')+'</div>';
  }
  c.innerHTML = '<div class="card"><div class="card-title"><span class="dot" style="background:var(--purple)"></span>Position Analysis '+ni+'</div>'
    +'<div class="fen-display">'+d.fen+'</div>'
    +'<div class="board-area"><div><div id="board-display">'+( d.board_svg||'')+'</div><div id="move-preview-area"></div></div>'
    +'<div class="suggestions-panel"><div style="font-size:0.85rem;font-weight:600;color:var(--text-bright);margin-bottom:4px">KG Suggestions <span style="font-size:0.75rem;color:var(--text-dim);font-weight:400">&mdash; click to preview on board</span></div>'+sh+hh+'</div></div></div>'
    +'<div class="card"><div class="card-title"><span class="dot"></span>Situation Features</div><div class="features-grid">'+fi+'</div>'
    +'<div style="margin-top:12px;font-size:0.75rem;color:var(--text-dim)"><strong>Signature:</strong> <code>'+(d.signature||'N/A')+'</code></div></div>';
  // Auto-preview first suggestion
  if (d.suggestions && d.suggestions.length) {
    const fr = c.querySelector('.sug-row');
    if (fr) previewMove(d.suggestions[0].move, fr);
  }
}

async function previewMove(uci, rowEl) {
  document.querySelectorAll('.sug-row').forEach(r => r.classList.remove('active'));
  if (rowEl) rowEl.classList.add('active');
  if (!currentTestFen) return;
  const pa = document.getElementById('move-preview-area');
  pa.innerHTML = '<div style="padding:12px;color:var(--text-dim);font-size:0.8rem">Loading preview...</div>';
  try {
    const r = await fetch('/api/move-preview', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({fen:currentTestFen, move:uci})});
    const d = await r.json();
    if (!d.valid) { pa.innerHTML = '<div style="padding:12px;color:var(--red)">'+d.error+'</div>'; return; }
    let badges = '';
    if (d.gives_check) badges += '<span class="badge badge-amber">Gives Check!</span> ';
    if (d.is_capture) badges += '<span class="badge badge-red">Capture</span> ';
    pa.innerHTML = '<div class="board-pair"><div class="board-col"><div class="board-label">Before <span class="move-tag">'+uci+'</span></div>'+d.before_svg+'</div>'
      +'<div class="board-col"><div class="board-label">After</div>'+d.after_svg+'<div class="move-info-badges">'+badges+'</div></div></div>'
      +'<div class="fen-display" style="font-size:0.72rem;margin-top:10px;color:var(--text-dim)">'+d.after_fen+'</div>';
    document.getElementById('board-display').innerHTML = d.before_svg;
  } catch(e) { pa.innerHTML = '<div style="padding:12px;color:var(--red)">'+e+'</div>'; }
}

// ==================== CONTROL ====================

async function startControl() {
  const btn = document.getElementById('btn-control');
  btn.disabled=true; btn.textContent='Running...';
  document.getElementById('ctrl-progress-card').style.display = 'block';
  document.getElementById('ctrl-results').innerHTML = '';
  document.getElementById('ctrl-log').innerHTML = '';
  const params = {
    positions: parseInt(document.getElementById('ctrl-positions').value),
    seed: parseInt(document.getElementById('ctrl-seed').value),
    oracle_depth: parseInt(document.getElementById('ctrl-depth').value),
  };
  try { await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params)}); }
  catch(e) { alert(e); btn.disabled=false; btn.textContent='Run Control Test'; return; }
  const poll = setInterval(async () => {
    try {
      const r = await fetch('/api/control/status'); const d = await r.json();
      const pct = d.total>0 ? Math.round(100*d.progress/d.total) : 0;
      document.getElementById('ctrl-pct').textContent = pct+'%';
      document.getElementById('ctrl-bar').style.width = pct+'%';
      document.getElementById('ctrl-count').textContent = d.progress+' / '+d.total;
      const logEl = document.getElementById('ctrl-log');
      (d.log||[]).forEach(line => {
        const div = document.createElement('div');
        div.className = line.includes('match')?'line-ok':line.includes('miss')?'line-warn':'line-info';
        div.textContent = line; logEl.appendChild(div);
      });
      if (d.log&&d.log.length) logEl.scrollTop = logEl.scrollHeight;
      if (!d.running) { clearInterval(poll); btn.disabled=false; btn.textContent='Run Control Test'; if(d.results) showCtrlResults(d.results); }
    } catch(e) {}
  }, 600);
}

function showCtrlResults(r) {
  const hp=Math.round(r.trained_hit_rate*100), mp=Math.round(r.trained_match_rate*100),
        t3=Math.round(r.trained_top3_rate*100), hep=Math.round(r.heuristic_match_rate*100),
        cm=r.trained_hits>0?Math.round(100*r.trained_matches/r.trained_hits):0;
  document.getElementById('ctrl-results').innerHTML =
    '<div class="card"><div class="card-title"><span class="dot" style="background:var(--amber)"></span>Results</div>'
    +'<div class="results-grid">'
    +'<div class="result-box"><div class="label">Tested</div><div class="value blue">'+r.positions_tested+'</div></div>'
    +'<div class="result-box"><div class="label">KG Hit Rate</div><div class="value '+(hp>50?'green':hp>20?'amber':'red')+'">'+hp+'%</div><div style="font-size:0.72rem;color:var(--text-dim);margin-top:4px">'+r.trained_hits+'/'+r.positions_tested+' had suggestions</div></div>'
    +'<div class="result-box"><div class="label">Top-1 Match</div><div class="value '+(mp>30?'green':mp>10?'amber':'red')+'">'+mp+'%</div></div>'
    +'<div class="result-box"><div class="label">Top-3 Match</div><div class="value '+(t3>40?'green':t3>15?'amber':'red')+'">'+t3+'%</div></div>'
    +'<div class="result-box"><div class="label">Heuristic</div><div class="value amber">'+hep+'%</div></div>'
    +'<div class="result-box"><div class="label">Untrained Hits</div><div class="value red">'+r.untrained_hits+'</div><div style="font-size:0.72rem;color:var(--text-dim);margin-top:4px">Control (always 0)</div></div>'
    +'</div>'
    +'<div style="padding:12px;background:var(--bg-raised);border-radius:8px;font-size:0.82rem;color:var(--text-dim);line-height:1.7">'
    +'<strong style="color:var(--text-bright)">Interpretation:</strong> '
    +'Trained KG recognized <strong>'+hp+'%</strong> of situations. '
    +'When it had a suggestion, top pick matched Stockfish <strong>'+cm+'%</strong>. '
    +'Untrained KG: <strong>0</strong> suggestions (control). '
    +'Heuristic baseline: <strong>'+hep+'%</strong>.</div></div>';
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

def get_kg() -> KG:
    kg = KG(DB_PATH)
    kg.init_schema()
    return kg

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/stats")
def api_stats():
    kg = get_kg(); stats = kg.stats(); kg.close()
    return jsonify(stats)

@app.route("/api/random-check")
def api_random_check():
    rng = random.Random(time.time())
    for _ in range(50):
        b = generate_random_check_position(rng, max_plies=80)
        if b and b.is_check() and list(b.legal_moves):
            return jsonify({"fen": b.fen()})
    return jsonify({"fen": None, "error": "Could not generate"})

@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json()
    fen = data.get("fen", "").strip()
    if not fen: return jsonify({"error": "No FEN"})
    try: b = chess.Board(fen)
    except Exception as e: return jsonify({"error": f"Invalid FEN: {e}"})
    kg = get_kg()
    result = query_position(kg, fen, limit=8)
    if b.is_check(): result["heuristic_moves"] = heuristic_escapes(b)
    kg.close()
    return jsonify(result)

@app.route("/api/move-preview", methods=["POST"])
def api_move_preview():
    data = request.get_json()
    fen = data.get("fen", "").strip()
    move_uci = data.get("move", "").strip()
    if not fen or not move_uci:
        return jsonify({"valid": False, "error": "Missing FEN or move"})
    return jsonify(render_board_with_move(fen, move_uci))

@app.route("/api/ollama/models", methods=["POST"])
def api_ollama_models():
    data = request.get_json() or {}
    host = data.get("host", OLLAMA_HOST).strip()
    return jsonify(OllamaClient.list_models(host))


@app.route("/api/ollama/debug", methods=["POST"])
def api_ollama_debug():
    """
    Two-part debug: first a simple ping to verify the model responds at all,
    then the actual chess prompt. Uses urllib (same as maze app) to eliminate
    any library differences.
    """
    import urllib.request
    import urllib.error
    data = request.get_json() or {}
    host = data.get("host", OLLAMA_HOST).strip().rstrip("/")
    model = data.get("model", "gpt-oss:20b")

    def ollama_generate(prompt_text, temp=0.3):
        """Exact same pattern as maze app's _ollama_generate."""
        url = f"{host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt_text,
            "stream": False,
            "options": {"temperature": float(temp)},
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw_bytes = resp.read().decode("utf-8")
            j = json.loads(raw_bytes)
            return {"response": (j.get("response") or "").strip(), "full": j, "error": None}
        except Exception as e:
            return {"response": "", "full": None, "error": str(e)}

    # ---- TEST 1: Simple ping ----
    ping_result = ollama_generate("What is 2+2? Answer in one word.", temp=0.1)

    # ---- TEST 2: Chess prompt ----
    import random as rnd
    rng = rnd.Random(42)
    b = None
    for _ in range(50):
        b = generate_random_check_position(rng, max_plies=60)
        if b and b.is_check() and list(b.legal_moves):
            break

    chess_result = {"response": "", "full": None, "error": "no test position"}
    prompt = ""
    legal_uci = []
    fen = ""
    if b and b.is_check():
        from engine import compute_situation_features
        feats = compute_situation_features(b)
        legal_uci = [mv.uci() for mv in b.legal_moves]
        fen = b.fen()
        prompt = OllamaClient._build_prompt(feats, legal_uci, fen)
        chess_result = ollama_generate(prompt)

    return jsonify({
        "model": model,
        "host": host,
        "ping": {
            "prompt": "What is 2+2? Answer in one word.",
            "response": ping_result["response"],
            "full_api_response": ping_result["full"],
            "error": ping_result["error"],
        },
        "chess": {
            "fen": fen,
            "legal_moves": legal_uci,
            "prompt": prompt,
            "response": chess_result["response"],
            "full_api_response": chess_result["full"],
            "error": chess_result["error"],
        },
    })

# ---- Training ----

@app.route("/api/train", methods=["POST"])
def api_train():
    if TRAIN_STATE["running"]:
        return jsonify({"error": "Training already in progress"})
    data = request.get_json()
    episodes = min(5000, max(10, int(data.get("episodes", 200))))
    seed = int(data.get("seed", 1))
    max_plies = int(data.get("max_plies", 60))
    oracle_depth = int(data.get("oracle_depth", 10))
    use_ollama = bool(data.get("use_ollama", False))
    ollama_model = str(data.get("ollama_model", "gpt-oss:20b"))
    ollama_host = str(data.get("ollama_host", OLLAMA_HOST))

    TRAIN_STATE.update(running=True, progress=0, total=episodes, log=[], summary=None)

    def run():
        kg = get_kg()
        def on_progress(ep, total, info):
            TRAIN_STATE["progress"] = ep
            st = info.get("status", "ok")
            if st == "ok":
                ch = info.get("chosen","?"); ob = info.get("oracle_best","?")
                m = "match!" if ch==ob else "miss"
                ln = info.get("ollama_proposed",0)
                ls = info.get("llm_status","")
                lt = f" [{ls}]" if ls else ""
                TRAIN_STATE["log"].append(f"[{ep}/{total}] {ch} vs oracle:{ob} -> {m} (boost={info.get('boost',0)}){lt}")
            elif ep % 5 == 0:
                TRAIN_STATE["log"].append(f"[{ep}/{total}] {st}")

        summary = train(
            kg, episodes, seed=seed, max_plies=max_plies,
            oracle_depth=oracle_depth, stockfish_path=STOCKFISH_PATH,
            use_ollama=use_ollama, ollama_host=ollama_host, ollama_model=ollama_model,
            progress_callback=on_progress,
        )
        kg.close()
        TRAIN_STATE["summary"] = summary
        TRAIN_STATE["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/train/status")
def api_train_status():
    log = TRAIN_STATE["log"][:]; TRAIN_STATE["log"] = []
    return jsonify({"running":TRAIN_STATE["running"],"progress":TRAIN_STATE["progress"],"total":TRAIN_STATE["total"],"log":log,"summary":TRAIN_STATE["summary"]})

# ---- Control ----

@app.route("/api/control", methods=["POST"])
def api_control():
    if CONTROL_STATE["running"]:
        return jsonify({"error": "Already running"})
    data = request.get_json()
    positions = min(500, max(10, int(data.get("positions", 50))))
    seed = int(data.get("seed", 999))
    oracle_depth = int(data.get("oracle_depth", 10))

    CONTROL_STATE.update(running=True, progress=0, total=positions, log=[], results=None)

    def run():
        kg = get_kg()
        def on_progress(i, total, info):
            CONTROL_STATE["progress"] = i
            if isinstance(info, dict) and info.get("status") != "skipped":
                ob = info.get("oracle_best","?"); tr = info.get("trained_suggestion","--")
                fd = info.get("trained_node_found",False)
                m = "match!" if tr==ob else "miss"
                s = f"found->{m}" if fd else "no-node"
                CONTROL_STATE["log"].append(f"[{i}/{total}] {info.get('check_type','?')} | oracle:{ob} trained:{tr} -> {s}")
        results = run_control_test(kg, positions, seed=seed, oracle_depth=oracle_depth, stockfish_path=STOCKFISH_PATH, progress_callback=on_progress)
        results.pop("details", None)
        kg.close()
        CONTROL_STATE["results"] = results
        CONTROL_STATE["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/control/status")
def api_control_status():
    log = CONTROL_STATE["log"][:]; CONTROL_STATE["log"] = []
    return jsonify({"running":CONTROL_STATE["running"],"progress":CONTROL_STATE["progress"],"total":CONTROL_STATE["total"],"log":log,"results":CONTROL_STATE["results"]})

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--db", default="kg.sqlite")
    args = p.parse_args()
    DB_PATH = args.db
    kg = get_kg(); kg.close()
    print(f"Escape-Check KG GUI: http://localhost:{args.port}")
    print(f"Database: {DB_PATH} | Stockfish: {STOCKFISH_PATH} | Ollama: {OLLAMA_HOST}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
