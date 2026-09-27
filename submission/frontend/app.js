"use strict";

/* ================================================================
   研迹 ScholarTrace — Frontend Controller
   ================================================================ */

const $ = (id) => document.getElementById(id);
const N = 5;

let _busy  = false;
let _timer = null;
let _historyCache = [];

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadHistory();

  $("search-btn").addEventListener("click", runSearch);
  $("query").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runSearch(); }
  });

  // 视图切换
  $("nav-search").addEventListener("click", () => switchView("search"));
  $("nav-dash").addEventListener("click",   () => switchView("dash"));

  $("query").focus();
});

// ── View Switch ──────────────────────────────────────────────────
function switchView(view) {
  const isSearch = view === "search";

  $("view-search").hidden = !isSearch;
  $("view-dash").hidden   = isSearch;

  $("nav-search").classList.toggle("active", isSearch);
  $("nav-dash").classList.toggle("active", !isSearch);
  $("nav-search").setAttribute("aria-pressed", isSearch);
  $("nav-dash").setAttribute("aria-pressed", !isSearch);

  if (!isSearch) renderDashboard();
}

// ── History ──────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const r = await fetch("/history?limit=30");
    if (!r.ok) return;
    _historyCache = await r.json();
    renderHistoryList(_historyCache.slice(0, 15));
  } catch { /* 静默 */ }
}

function renderHistoryList(items) {
  const el = $("history-list");
  if (!items?.length) {
    el.innerHTML = `<p class="hist-empty">尚無查詢記錄<br><em>No search history</em></p>`;
    return;
  }
  el.innerHTML = "";
  items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "hist-item";
    div.setAttribute("role", "button");
    div.innerHTML = `
      <span class="hist-q" title="${esc(item.query)}">${esc(item.query)}</span>
      <div class="hist-meta">
        <span class="hist-mode">${esc(item.mode)}</span>
        <span>${item.result_count} 篇</span>
      </div>`;
    div.addEventListener("click", () => {
      if (_busy) return;
      switchView("search");
      $("query").value = item.query;
      $("mode").value  = item.mode;
      runSearch();
    });
    el.appendChild(div);
  });
}

// ── Dashboard ────────────────────────────────────────────────────
function renderDashboard() {
  const items = _historyCache;
  if (!items.length) {
    $("sess-queries").textContent = "0";
    $("sess-papers").textContent  = "0";
    $("sess-mode").textContent    = "—";
    $("mode-breakdown").innerHTML = "";
    return;
  }

  const totalPapers = items.reduce((s, h) => s + (h.result_count || 0), 0);
  $("sess-queries").textContent = items.length;
  $("sess-papers").textContent  = totalPapers.toLocaleString();

  // 模式分布
  const modeCounts = {};
  items.forEach(h => { modeCounts[h.mode] = (modeCounts[h.mode] || 0) + 1; });
  const topMode = Object.entries(modeCounts).sort((a, b) => b[1] - a[1])[0];
  $("sess-mode").textContent = topMode ? topMode[0].toUpperCase() : "—";

  // 模式分布 chips
  const breakdown = $("mode-breakdown");
  breakdown.innerHTML = "";
  Object.entries(modeCounts)
    .sort((a, b) => b[1] - a[1])
    .forEach(([mode, count]) => {
      const chip = document.createElement("span");
      chip.className = "mode-chip";
      chip.innerHTML = `<span class="mode-chip-label">${esc(mode)}</span><span class="mode-chip-count">${count} 次</span>`;
      breakdown.appendChild(chip);
    });
}

// ── Search ───────────────────────────────────────────────────────
async function runSearch() {
  const q = $("query").value.trim();
  if (!q || _busy) return;
  _busy = true;

  const btn = $("search-btn");
  btn.disabled = true;
  $("btn-label").textContent = "處理中…";

  // 折叠标题，紧凑布局
  $("main-inner").classList.add("compact");

  resetView();
  startPipeline();

  try {
    const resp = await fetch("/search", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query:     q,
        mode:      $("mode").value,
        top_k:     20,
        summarize: $("summarize").checked,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);

    finishPipeline(true);
    renderResults(data);
    await loadHistory();
  } catch (err) {
    finishPipeline(false);
    showError(err.message);
  } finally {
    _busy = false;
    btn.disabled = false;
    $("btn-label").textContent = "檢索";
  }
}

function resetView() {
  $("result-list").innerHTML = "";
  $("results-wrap").hidden   = true;
  $("telemetry").hidden      = true;
  $("summary-panel").hidden  = true;
  $("graph-section").hidden  = true;

  const ph = $("placeholder");
  ph.hidden = false;
  ph.querySelector(".ph-t").textContent = "就緒 · Ready";
  ph.querySelector(".ph-b").textContent = "輸入查詢並選擇引擎模式以啟動全鏈路文獻檢索";
}

// ── Pipeline ─────────────────────────────────────────────────────
const pn = (i) => $(`pn-${i}`);

function startPipeline() {
  clearInterval(_timer);
  $("pipeline").hidden = false;
  $("pipe-status").textContent = "處理中… Processing";
  $("pipe-fill").style.transform = "scaleX(0)";
  for (let i = 0; i < N; i++) pn(i)?.classList.remove("active", "done");

  activate(0);
  let cur = 0;
  _timer = setInterval(() => {
    if (cur < N - 2) {
      done(cur);
      activate(++cur);
      $("pipe-fill").style.transform = `scaleX(${cur / (N - 1)})`;
    }
  }, 1050);
}

function activate(i) { pn(i)?.classList.add("active"); }
function done(i)     { const n = pn(i); n?.classList.remove("active"); n?.classList.add("done"); }

function finishPipeline(ok) {
  clearInterval(_timer);
  if (ok) {
    for (let i = 0; i < N; i++) done(i);
    $("pipe-fill").style.transform = "scaleX(1)";
    $("pipe-status").textContent = "完成 · Completed";
    setTimeout(() => { $("pipeline").hidden = true; }, 1200);
  } else {
    $("pipe-status").textContent = "失敗 · Failed";
  }
}

// ── Results ──────────────────────────────────────────────────────
function renderResults(data) {
  const tm = data.telemetry || {};

  $("tel-count").textContent   = `${data.results.length} 篇`;
  $("tel-llm").textContent     = `${tm.llm_calls ?? "—"} calls`;
  $("tel-tokens").textContent  = typeof tm.total_tokens === "number"
    ? tm.total_tokens.toLocaleString() + " tok" : "—";
  $("tel-latency").textContent = `${tm.endpoint_latency_ms ?? "—"} ms`;
  $("tel-trace").textContent   = (data.traces || [])
    .map(t => `R${t.round}:${t.api.toUpperCase()}`).join(" → ");
  $("telemetry").hidden = false;

  if (!data.results.length) {
    const ph = $("placeholder");
    ph.hidden = false;
    ph.querySelector(".ph-t").textContent = "無結果 · No Results";
    ph.querySelector(".ph-b").textContent = "未找到匹配文獻，請調整查詢或切換 FULL 全鏈路模式";
    return;
  }

  $("placeholder").hidden  = true;
  $("results-wrap").hidden = false;
  $("result-count").textContent = `${data.results.length} 篇`;

  if (data.summary?.query_summary) {
    $("query-summary").textContent = data.summary.query_summary;
    const gc = $("groups");
    gc.innerHTML = "";
    (data.summary.groups || []).forEach((g) => {
      const c = document.createElement("span");
      c.className = "sum-chip";
      c.innerHTML = `${esc(g.name)}<em>${g.paper_ids.length} 篇</em>`;
      gc.appendChild(c);
    });
    $("overall").textContent  = data.summary.overall_summary || "";
    $("summary-panel").hidden = false;
  }

  const list = $("result-list");
  list.innerHTML = "";

  data.results.forEach((r, i) => {
    const li = document.createElement("li");
    li.className = "result-item";
    li.style.animationDelay = `${i * 38}ms`;

    const authors = (r.authors || []).slice(0, 3).join(", ")
      + (r.authors?.length > 3 ? " et al." : "");
    const covHtml = Object.keys(r.constraint_coverage || {})
      .map(k => `<span class="cov-tag">${esc(k)}</span>`).join("");

    li.innerHTML = `
      <div class="r-top">
        <span class="r-num">${i + 1}</span>
        <span class="r-label ${r.label}">${r.label}</span>
        <a class="r-title"
           href="https://openalex.org/${esc(r.paper_id)}"
           target="_blank" rel="noopener">${esc(r.title)}</a>
        <span class="r-score">${Number(r.score).toFixed(3)}</span>
      </div>
      <div class="r-meta">
        ${authors ? `<span>${esc(authors)}</span>` : ""}
        ${r.venue  ? `<span class="r-ms">·</span><span>${esc(r.venue)}</span>` : ""}
        ${r.year   ? `<span class="r-ms">·</span><span>${r.year}</span>` : ""}
        <span class="r-ms">·</span>
        <span class="r-pid">${esc(r.paper_id)}</span>
      </div>
      ${r.reason  ? `<div class="r-reason">${esc(r.reason)}</div>`  : ""}
      ${covHtml   ? `<div class="r-cov">${covHtml}</div>`            : ""}`;
    list.appendChild(li);
  });

  if (data.summary?.relation_graph?.edges?.length > 0) {
    $("graph-section").hidden = false;
    setTimeout(() => drawGraph(data.summary.relation_graph), 80);
  }
}

function showError(msg) {
  const ph = $("placeholder");
  ph.hidden = false;
  ph.querySelector(".ph-t").textContent = "檢索失敗";
  ph.querySelector(".ph-b").innerHTML   = `<span style="color:#B91C1C">${esc(msg)}</span>`;
}

// ── Graph ────────────────────────────────────────────────────────
function drawGraph(graph) {
  const svg  = $("graph-svg");
  const tip  = $("graph-tip");
  svg.innerHTML = "";

  const W     = svg.clientWidth || 640;
  const H     = 280;
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  if (!nodes.length) return;

  const cx = {}, cy = {};
  nodes.forEach((node, i) => {
    const a = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
    const r = i % 2 === 0 ? 100 : 65;
    cx[node.id] = W / 2 + r * Math.cos(a);
    cy[node.id] = H / 2 + r * Math.sin(a);
  });

  edges.forEach((e) => {
    if (cx[e.source] == null) return;
    const line = svgEl("line");
    line.setAttribute("x1", cx[e.source]); line.setAttribute("y1", cy[e.source]);
    line.setAttribute("x2", cx[e.target]); line.setAttribute("y2", cy[e.target]);
    const ref = e.kind === "reference";
    line.setAttribute("stroke",       ref ? "#CAC6BC" : "#1B3A6B");
    line.setAttribute("stroke-width", ref ? "1" : "1.2");
    line.setAttribute("stroke-dasharray", ref ? "4 3" : "");
    line.setAttribute("opacity", "0.5");
    svg.appendChild(line);
  });

  nodes.forEach((node) => {
    const g = svgEl("g");
    g.style.cursor = "pointer";

    const c = svgEl("circle");
    c.setAttribute("cx", cx[node.id]); c.setAttribute("cy", cy[node.id]);
    c.setAttribute("r", "5");
    c.setAttribute("fill", "#FFFFFF"); c.setAttribute("stroke", "#1B3A6B");
    c.setAttribute("stroke-width", "1.5");

    const t = svgEl("text");
    t.setAttribute("x", cx[node.id] + 9); t.setAttribute("y", cy[node.id] + 3.5);
    t.setAttribute("font-size", "9.5"); t.setAttribute("fill", "#9C9891");
    t.setAttribute("font-family", "system-ui, sans-serif");
    t.textContent = (node.title || node.id).slice(0, 18);

    g.appendChild(c); g.appendChild(t);

    g.addEventListener("mouseenter", () => {
      c.setAttribute("r", "7.5"); c.setAttribute("fill", "#1B3A6B");
      t.setAttribute("fill", "#1A1917");
      tip.textContent   = node.title || node.id;
      tip.style.opacity = "1";
      tip.style.left    = `${cx[node.id] + 12}px`;
      tip.style.top     = `${cy[node.id] - 30}px`;
    });
    g.addEventListener("mouseleave", () => {
      c.setAttribute("r", "5"); c.setAttribute("fill", "#FFFFFF");
      t.setAttribute("fill", "#9C9891");
      tip.style.opacity = "0";
    });
    svg.appendChild(g);
  });
}

const svgEl = (tag) => document.createElementNS("http://www.w3.org/2000/svg", tag);

// ── Escape ───────────────────────────────────────────────────────
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
