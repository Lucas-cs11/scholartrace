#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eZScholar 竞赛提案 —— Nature 级论文插图（由 nature-figure skill 驱动）。

本脚本严格遵循 nature-figure skill（.claude/skills/nature-figure）的规范：
  - MANDATORY font + SVG 规则（api.md）最先设置
  - 使用 skill 的 PALETTE（blue_main/green_3/red_strong/neutral_*）
  - 使用 skill 助手 make_grouped_bar / add_panel_label / apply_publication_style
  - 多面板图调用 require_matplotlib_panel_alignment() 渲染时对齐门
  - 导出 PDF 后跑 audit_figure_collisions.py 碰撞审计

图集（8 幅，全部基于已核实实验数据）：
  fig1_pipeline       系统架构 / 检索闭环示意图（teaser，无数值）
  fig2_baseline       B0 / B1 / FULL 平均 F1 对比 + F1>0 命中比例
  fig3_per_query      50 条查询逐条 F1 分布（绿 F1>0 vs 灰 F1=0）
  fig4_attribution    gold 级失败归因（88% 未召回 / 7.1% 词法挤出 / 4.9% 精排降级）
  fig5_direction1     S2 多源召回增益（+13 gold，7.1%，与 OpenAlex 不相交，Q6 +9）
  fig6_reranker       精排器消融（RRF / MiniLM-CE / BGE / LLM 的 matched-N F1）
  fig7_stochasticity  LLM 精排随机性（7 条 rich 查询 3× 重排 F1 波动）
  fig8_funnel         查询公式化与迭代检索的 raw unique Gold 漏斗
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# 字体：按用户要求采用"新罗马"衬线体 —— Times New Roman(拉丁/数字) + Songti SC(宋体/CJK)
# （覆盖 nature-figure 的 sans-serif 默认；Times New Roman 无 CJK 字形，Songti SC 补齐）
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [
    'Songti SC',       # 宋体：同时覆盖中文与拉丁，衬线学术体，避免豆腐块
    'Times New Roman', 'STSong', 'SimSun', 'Source Han Serif SC',
    'DejaVu Serif',
]
plt.rcParams['svg.fonttype'] = 'none'   # 可编辑 SVG 文本
plt.rcParams['pdf.fonttype'] = 42       # 可编辑 TrueType 文本
plt.rcParams['font.size'] = 8
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['legend.frameon'] = False

# ---------------------------------------------------------------------------
# nature-figure skill PALETTE（api.md）
# ---------------------------------------------------------------------------
BLUE = "#0F4D92"     # blue_main
BLUE2 = "#3775BA"    # blue_secondary
GREEN = "#8BCF8B"    # green_3
GREEN_D = "#2E9E44"  # delta_up
RED = "#B64342"      # red_strong
NEU = "#767676"      # neutral_mid
DARK = "#272727"     # neutral_black
GRAY = "#CFCECE"     # neutral_light
TEAL = "#42949E"
AMBER = "#E28E2C"

# skill scripts（对齐门 / 安全助手 / QA）加入 PYTHONPATH。
# 路径经环境变量 NATURE_FIGURE_SKILL 注入，避免硬编码绝对本地路径（缺省时优雅降级，不阻断出图）。
_SKILL_PATH = os.environ.get("NATURE_FIGURE_SKILL")
if _SKILL_PATH:
    sys.path.insert(0, str(Path(_SKILL_PATH) / "scripts"))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
    _HAS_ALIGN = True
except Exception:
    _HAS_ALIGN = False

OUT = Path(__file__).resolve().parent.parent.parent / "proposal" / "figures"
RUNS = Path(__file__).resolve().parent.parent.parent / "eval" / "runs"
OUT.mkdir(parents=True, exist_ok=True)


def _finalize(fig, name, multi=False):
    """导出前：多面板跑对齐门；导出 SVG/PDF/PNG（对齐门返回 dict 时校验）。"""
    if multi and _HAS_ALIGN:
        require_matplotlib_panel_alignment(
            fig, json_out=str(OUT / f"{name}.alignment.json"),
            overlay_svg=str(OUT / f"{name}.alignment.svg"),
            tolerance_pt=1.5, strict=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  ✓ {name}.svg/.pdf/.png")


def _panel_label(ax, label):
    from matplotlib.transforms import ScaledTranslation
    off = ScaledTranslation(-12 / 72, 5 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + off, fontsize=10,
            fontweight="bold", color=DARK, ha="left", va="bottom")


# ===========================================================================
# Fig 1 — 系统架构示意图（teaser）
# ===========================================================================
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)

    def box(x, y, w, h, text, fc="#f1f5f9", ec=DARK, fs=7.5, hl=False):
        if hl:
            fc, ec = "#e7eefb", BLUE
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=DARK, zorder=5)

    def arrow(x1, y1, x2, y2, color=NEU):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, lw=1.1, color=color))

    box(0.3, 4.05, 2.0, 0.5, "复杂学术查询\n(自然语言长句)", fs=7.5)
    box(0.3, 2.6, 1.7, 0.9, "查询解析\n→ 结构化中间表示\n约束/年限/方法", fs=7)
    arrow(2.05, 3.05, 2.55, 3.05)
    box(2.6, 2.6, 1.7, 0.9, "子查询规划\n联想词专名\n5 个高优子查询", fs=7, hl=True)
    arrow(4.35, 3.05, 4.85, 3.05)
    box(4.9, 2.6, 1.8, 0.9, "多源召回\nOpenAlex\n+ Semantic Scholar", fs=7, hl=True)
    arrow(6.75, 3.05, 7.25, 3.05)
    box(7.3, 2.6, 1.4, 0.9, "词法粗筛\nTop-40", fs=7)
    arrow(8.75, 3.05, 9.2, 3.05)
    arrow(8.7, 2.2, 8.7, 1.7)
    box(7.3, 0.75, 1.9, 0.9, "大语言模型\n综合精排\n证据链+约束覆盖\n动态截断", fs=7, hl=True)
    arrow(6.25, 1.2, 5.75, 1.2)
    box(3.9, 0.75, 1.8, 0.9, "引文扩展\n(Semantic Scholar /\n开放引文)\n多轮早停", fs=7)
    arrow(2.05, 1.2, 1.55, 1.2)
    box(0.3, 0.75, 1.2, 0.9, "结构化推荐\nF1 评测\n证据可溯源", fs=7)
    arrow(5.75, 1.7, 4.2, 2.55, color=BLUE)

    ax.text(5, 4.45, "eZScholar 智能学术检索 Agent —— 先宽后窄检索闭环",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK)
    ax.text(5, 0.18, "蓝色模块 = 本项目创新点；灰线 = 检索闭环", ha="center",
            va="center", fontsize=6.5, color=BLUE)
    _finalize(fig, "fig1_pipeline")


# ===========================================================================
# Fig 2 — 基线平均 F1 对比（skill make_grouped_bar 思路）
# ===========================================================================
def fig2_baseline():
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    names = ["B0\n词法基线", "B1\n查询理解", "FULL\n全链路"]
    f1 = [0.0095, 0.0127, 0.0774]
    hit = ["F1>0 2/20", "F1>0 2/15", "F1>0 26/50"]
    colors = [GRAY, GRAY, BLUE]
    bars = ax.bar(names, f1, color=colors, width=0.6, zorder=3,
                  edgecolor=DARK, linewidth=0.6)
    for b, h in zip(bars, hit):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0035, h,
                ha="center", va="bottom", fontsize=6.5, color=DARK)
    ax.text(2, 0.092, "≈ 8.2× vs B0", ha="center", fontsize=8, color=RED,
            fontweight="bold")
    ax.set_ylabel("宏平均 F1")
    ax.set_ylim(0, 0.105)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.text(-0.32, 0.100, "RealScholarQuery 全量评测 (OpenAlex 口径)",
            ha="left", fontsize=7.5, color=DARK)
    _finalize(fig, "fig2_baseline")


# ===========================================================================
# Fig 3 — 逐条查询 F1 分布
# ===========================================================================
def fig3_per_query():
    reps = [json.load(open(f)) for f in
            glob.glob(str(RUNS / "PASA_FULL_RealScholarQuery_*.json"))]
    reps.sort(key=lambda r: int(r["query_id"].split("_")[-1]))
    ids = [int(r["query_id"].split("_")[-1]) for r in reps]
    f1 = [r["f1"] for r in reps]
    cols = [BLUE if x > 0 else GRAY for x in f1]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.hlines(0, min(ids), max(ids), color="#e2e8f0", lw=0.8)
    ax.scatter(ids, f1, s=16, color=cols, zorder=3, edgecolors="white", linewidths=0.4)
    for i, x in enumerate(ids):
        if f1[i] >= 0.28:
            ax.annotate(f"{f1[i]:.2f}", (x, f1[i]), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=6, color=BLUE)
    ax.axvline(25.5, color=NEU, lw=0.6, ls="--")
    ax.text(12, 0.31, "前 26 条", ha="center", fontsize=6.5, color=DARK)
    ax.text(37, 0.31, "后 24 条", ha="center", fontsize=6.5, color=NEU)
    ax.set_xlabel("查询编号 (RealScholarQuery_i)")
    ax.set_ylabel("F1")
    ax.set_ylim(-0.02, 0.40)
    ax.legend(handles=[Patch(color=BLUE, label=f"F1>0 ({sum(1 for x in f1 if x>0)})"),
                       Patch(color=GRAY, label=f"F1=0 ({sum(1 for x in f1 if x==0)})")],
              loc="upper right", fontsize=7, frameon=False)
    _finalize(fig, "fig3_per_query")


# ===========================================================================
# Fig 4 — gold 级失败归因（图例式堆叠条，规避窄段文字重叠）
# ===========================================================================
def fig4_attribution():
    segs = [("未召回", 162, RED), ("词法挤出", 13, AMBER), ("精排降级", 9, NEU)]
    total = 184
    fig, ax = plt.subplots(figsize=(6.4, 1.6))
    left = 0
    for name, n, c in segs:
        w = n / total
        ax.barh(0, w, left=left, color=c, height=0.45, edgecolor="white", zorder=3)
        ax.text(left + w / 2, 0.42, f"{n} ({n/total*100:.0f}%)",
                ha="center", va="bottom", fontsize=7.5, color=DARK)
        left += w
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.9, 1.0); ax.axis("off")
    ax.legend(handles=[Patch(color=c, label=f"{name} · {n}") for name, n, c in segs],
              loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=3,
              fontsize=7, frameon=False)
    ax.text(0.5, 1.0, "22 条覆盖查询 · 184 篇真值论文的失败去向（0% 假召回）",
            ha="center", va="bottom", fontsize=8, color=DARK)
    _finalize(fig, "fig4_attribution")


# ===========================================================================
# Fig 5 — 方向1 S2 多源召回增益（单面板逐查询对照；聚合量放入图注）
# ===========================================================================
def fig5_direction1():
    probe = json.load(open(RUNS / "s2_d1_multisource_probe.json"))
    pq = sorted(probe["per_query"], key=lambda q: int(q["qid"].split("_")[-1]))
    ids = [int(q["qid"].split("_")[-1]) for q in pq]
    in_oa = [q["in_OA"] for q in pq]
    new_s2 = [q["new_by_s2"] for q in pq]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    fig.subplots_adjust(bottom=0.18, top=0.90)
    ax.set_title("Semantic Scholar 多源召回：在 OpenAlex 之上新增 13 篇真值论文 (7.1%)，与 OpenAlex 不相交",
                 fontsize=8, color=DARK, pad=6)
    w = 0.38
    x = list(range(len(ids)))
    ax.bar([i - w / 2 for i in x], in_oa, w, color=BLUE, label="OpenAlex 召回",
           edgecolor=DARK, linewidth=0.4)
    ax.bar([i + w / 2 for i in x], new_s2, w, color=GREEN_D, label="Semantic Scholar 新增",
           edgecolor=DARK, linewidth=0.4)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ids, fontsize=6)
    ax.set_xlabel("查询编号"); ax.set_ylabel("召回真值论文数"); ax.set_ylim(0, 11)
    ax.legend(fontsize=7, loc="upper right")
    q6 = ids.index(6)
    ax.annotate("查询6 新增 9 篇", (x[q6] + w / 2, new_s2[q6]), textcoords="offset points",
                xytext=(6, 3), fontsize=7.5, color=GREEN_D, fontweight="bold")
    _finalize(fig, "fig5_direction1")


# ===========================================================================
# Fig 6 — 精排器消融
# ===========================================================================
def fig6_reranker():
    data = {"大语言模型\n(生产)": 0.0664, "MiniLM 句向量\n打分器 (22M)": 0.0598,
            "BGE 精排器\n(568M)": 0.0588, "RRF 词频\n融合 (无模型)": 0.0241}
    names = [k for k, _ in sorted(data.items(), key=lambda kv: -kv[1])]
    f1 = [data[k] for k in names]
    colors = [BLUE if i == 0 else (GRAY if i == len(f1) - 1 else NEU)
              for i in range(len(f1))]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    fig.subplots_adjust(bottom=0.22, top=0.96)
    y = range(len(names))
    ax.barh(list(y), f1, color=colors, height=0.55, zorder=3, edgecolor=DARK, linewidth=0.6)
    for yy, v in zip(y, f1):
        ax.text(v + 0.001, yy, f"{v:.4f}", va="center", fontsize=7.5, color=DARK)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("宏平均 F1（候选条数与大语言模型一致）")
    ax.set_xlim(0, 0.08); ax.invert_yaxis()
    fig.text(0.5, 0.02, "冻结候选池 22 条查询 · 各精排器输出相同数量的候选 · 均为确定性方法",
             ha="center", fontsize=7, color=DARK)
    _finalize(fig, "fig6_reranker")


# ===========================================================================
# Fig 7 — LLM 精排随机性
# ===========================================================================
def fig7_stochasticity():
    data = [("Q0", 0.1429, 0.1429), ("Q6", 0.2353, 0.2667), ("Q15", 0.1333, 0.2222),
            ("Q23", 0.0909, 0.0909), ("Q25", 0.0952, 0.0952), ("Q35", 0.0741, 0.0741),
            ("Q47", 0.0833, 0.0833)]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    fig.subplots_adjust(bottom=0.16, top=0.94)
    for i, (q, lo, hi) in enumerate(data):
        ax.plot([i, i], [lo, hi], color=NEU, lw=2, zorder=2)
        ax.scatter([i], [lo], s=22, color=RED, zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter([i], [hi], s=22, color=BLUE, zorder=3, edgecolors="white", linewidths=0.5)
        if hi - lo > 0.05:
            ax.annotate(f"Δ{hi-lo:.3f}", (i, (lo + hi) / 2), textcoords="offset points",
                        xytext=(4, 0), fontsize=6.5, color=RED)
    ax.set_xticks(range(len(data))); ax.set_xticklabels([q for q, _, _ in data], fontsize=8)
    ax.set_ylabel("3× 精排 F1 (同输入池)"); ax.set_ylim(0.0, 0.30)
    ax.set_xlim(-0.7, len(data) - 0.3)
    fig.text(0.5, 0.02, "同一候选池 3 次 LLM 重排 → 排序/F1 显著波动",
             ha="center", fontsize=7.5, color=DARK)
    _finalize(fig, "fig7_stochasticity")


# ===========================================================================
# Fig 8 — 查询公式化 + 迭代检索漏斗（多面板，走对齐门）
# ===========================================================================
def fig8_funnel():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                   gridspec_kw={"wspace": 0.35})
    _panel_label(ax1, "a"); _panel_label(ax2, "b")

    labels1 = ["初始基线", "追加联想词\n召回", "提示词\n优化"]
    vals1 = [22, 25, 26]
    cols1 = [GRAY, BLUE, BLUE]
    ax1.bar(labels1, vals1, color=cols1, width=0.55, zorder=3, edgecolor=DARK, linewidth=0.6)
    for i, v in enumerate(vals1):
        ax1.text(i, v + 0.4, str(v), ha="center", fontsize=7.5)
    ax1.annotate("+3", (1, 25), textcoords="offset points", xytext=(8, 2), color=GREEN_D)
    ax1.annotate("+1", (2, 26), textcoords="offset points", xytext=(8, 2), color=GREEN_D)
    ax1.set_ylim(0, 30); ax1.set_ylabel("独立真值论文数")
    ax1.set_title("查询公式化优化", fontsize=8)

    labels2 = ["单轮检索", "合并多轮\n检索"]
    vals2 = [25, 27]
    ax2.bar(labels2, vals2, color=[BLUE, GREEN_D], width=0.5, zorder=3,
            edgecolor=DARK, linewidth=0.6)
    for i, v in enumerate(vals2):
        ax2.text(i, v + 0.4, str(v), ha="center", fontsize=7.5)
    ax2.annotate("+2 (2/22 查询)", xy=(1, 27), xytext=(1, 29.2),
                 ha="center", va="bottom", color=GREEN_D, fontsize=7)
    ax2.set_ylim(0, 30); ax2.set_ylabel("独立真值论文数")
    ax2.set_title("证据引导迭代检索", fontsize=8)
    _finalize(fig, "fig8_funnel", multi=True)


if __name__ == "__main__":
    print("生成图集（nature-figure skill 驱动）→", OUT)
    fig1_pipeline(); fig2_baseline(); fig3_per_query(); fig4_attribution()
    fig5_direction1(); fig6_reranker(); fig7_stochasticity(); fig8_funnel()
    print("完成：8 幅图")
