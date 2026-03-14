#!/usr/bin/env python3
"""TENG 三分类（walk/run/jump）示例：纯标准库实现，无需第三方依赖。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

LABELS = ["walk", "run", "jump"]


@dataclass
class SampleWindow:
    label: str
    signal: List[float]


def segment_signal(signal: Sequence[float], label: str, window_size: int = 128, step: int = 64) -> List[SampleWindow]:
    if window_size <= 0:
        raise ValueError("window_size 必须大于 0")
    if step <= 0:
        raise ValueError("step 必须大于 0")
    return [
        SampleWindow(label, list(signal[i : i + window_size]))
        for i in range(0, len(signal) - window_size + 1, step)
    ]


def read_csv_column(csv_path: Path, col_name: str) -> List[float]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if col_name not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} 缺少列: {col_name}")
        out: List[float] = []
        for row in reader:
            try:
                out.append(float(row[col_name]))
            except (TypeError, ValueError):
                continue
        return out


def load_real_dataset(data_dir: Path, signal_column: str) -> List[SampleWindow]:
    windows: List[SampleWindow] = []
    for label in LABELS:
        class_dir = data_dir / label
        if not class_dir.exists():
            continue
        for csv_file in sorted(class_dir.glob("*.csv")):
            sig = read_csv_column(csv_file, signal_column)
            if len(sig) >= 128:
                windows.extend(segment_signal(sig, label))
    return windows


def _gauss(rng: random.Random, n: int, sigma: float) -> List[float]:
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def generate_synthetic_dataset(n_windows_per_class: int = 400, window_size: int = 128, seed: int = 42) -> List[SampleWindow]:
    rng = random.Random(seed)
    windows: List[SampleWindow] = []
    t = [2 * math.pi * i / (window_size - 1) for i in range(window_size)]

    for _ in range(n_windows_per_class):  # walk
        f, a, p = rng.uniform(1.2, 1.8), rng.uniform(0.5, 1.0), rng.uniform(0, 2 * math.pi)
        n = _gauss(rng, window_size, 0.14)
        sig = [a * math.sin(f * x + p) + n[i] for i, x in enumerate(t)]
        windows.append(SampleWindow("walk", sig))

    for _ in range(n_windows_per_class):  # run
        f, a, p = rng.uniform(2.4, 3.4), rng.uniform(0.9, 1.5), rng.uniform(0, 2 * math.pi)
        n = _gauss(rng, window_size, 0.18)
        sig = [a * math.sin(f * x + p) + 0.2 * math.sin(2 * f * x) + n[i] for i, x in enumerate(t)]
        windows.append(SampleWindow("run", sig))

    for _ in range(n_windows_per_class):  # jump
        f = rng.uniform(0.7, 1.1)
        n = _gauss(rng, window_size, 0.12)
        sig = [0.2 * math.sin(f * x) + n[i] for i, x in enumerate(t)]
        for _ in range(rng.randint(2, 4)):
            pos = rng.randint(0, window_size - 1)
            width = rng.randint(2, 6)
            height = rng.uniform(1.5, 3.2)
            for i in range(max(0, pos - width), min(window_size, pos + width + 1)):
                rel = (i - pos) / max(width, 1)
                sig[i] += height * math.exp(-8 * rel * rel)
        windows.append(SampleWindow("jump", sig))

    return windows


def dft_power(signal: Sequence[float], k_max: int = 16) -> List[float]:
    n = len(signal)
    powers: List[float] = []
    for k in range(1, min(k_max, n // 2) + 1):
        re = 0.0
        im = 0.0
        for i, x in enumerate(signal):
            angle = -2 * math.pi * k * i / n
            re += x * math.cos(angle)
            im += x * math.sin(angle)
        powers.append(re * re + im * im)
    return powers


def extract_features(sig: Sequence[float]) -> List[float]:
    s = list(sig)
    n = len(s)
    if n == 0:
        raise ValueError("空信号无法提取特征")
    mean = statistics.fmean(s)
    var = statistics.fmean([(x - mean) ** 2 for x in s])
    std = math.sqrt(var)
    abs_s = [abs(x) for x in s]
    rms = math.sqrt(statistics.fmean([x * x for x in s]))
    peak = max(s) - min(s)
    zcr = sum(1 for i in range(1, n) if s[i - 1] * s[i] < 0) / max(n - 1, 1)

    sorted_s = sorted(s)
    q25 = sorted_s[int(0.25 * (n - 1))]
    q50 = sorted_s[int(0.50 * (n - 1))]
    q75 = sorted_s[int(0.75 * (n - 1))]

    power = dft_power(s, k_max=16)
    total = sum(power) + 1e-12
    centroid = sum((i + 1) * p for i, p in enumerate(power)) / total
    probs = [p / total for p in power]
    entropy = -sum(p * math.log2(p + 1e-12) for p in probs)

    return [
        mean,
        std,
        rms,
        statistics.fmean(abs_s),
        peak,
        zcr,
        (max(abs_s) + 1e-12) / (rms + 1e-12),
        q25,
        q50,
        q75,
        centroid,
        entropy,
    ]


class GaussianNB:
    def __init__(self) -> None:
        self.classes: List[str] = []
        self.priors: Dict[str, float] = {}
        self.means: Dict[str, List[float]] = {}
        self.vars: Dict[str, List[float]] = {}

    def fit(self, x: List[List[float]], y: List[str]) -> None:
        if not x or not y:
            raise ValueError("训练数据不能为空")
        if len(x) != len(y):
            raise ValueError("x 与 y 长度不一致")
        self.classes = sorted(set(y))
        for c in self.classes:
            idx = [i for i, label in enumerate(y) if label == c]
            xc = [x[i] for i in idx]
            self.priors[c] = len(xc) / len(x)
            cols = list(zip(*xc))
            self.means[c] = [statistics.fmean(col) for col in cols]
            self.vars[c] = [statistics.fmean([(v - m) ** 2 for v in col]) + 1e-9 for col, m in zip(cols, self.means[c])]

    def predict_one(self, f: List[float]) -> str:
        best_c, best_logp = None, float("-inf")
        for c in self.classes:
            logp = math.log(self.priors[c] + 1e-12)
            for v, mu, var in zip(f, self.means[c], self.vars[c]):
                logp += -0.5 * math.log(2 * math.pi * var) - ((v - mu) ** 2) / (2 * var)
            if logp > best_logp:
                best_c, best_logp = c, logp
        return best_c or self.classes[0]

    def predict(self, x: List[List[float]]) -> List[str]:
        return [self.predict_one(f) for f in x]

    def predict_proba(self, x: List[List[float]]) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for f in x:
            logps: Dict[str, float] = {}
            for c in self.classes:
                logp = math.log(self.priors[c] + 1e-12)
                for v, mu, var in zip(f, self.means[c], self.vars[c]):
                    logp += -0.5 * math.log(2 * math.pi * var) - ((v - mu) ** 2) / (2 * var)
                logps[c] = logp

            max_logp = max(logps.values())
            exps = {c: math.exp(lp - max_logp) for c, lp in logps.items()}
            s = sum(exps.values()) or 1.0
            out.append({c: v / s for c, v in exps.items()})
        return out


def train_test_split(x: List[List[float]], y: List[str], test_ratio: float = 0.25, seed: int = 42):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio 必须在 (0,1) 区间")
    if len(x) != len(y):
        raise ValueError("x 与 y 长度不一致")
    rng = random.Random(seed)
    by_class: Dict[str, List[int]] = {c: [] for c in sorted(set(y))}
    for i, c in enumerate(y):
        by_class[c].append(i)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for idxs in by_class.values():
        if len(idxs) < 2:
            raise ValueError("每个类别至少需要 2 个样本用于划分训练/测试集")
        rng.shuffle(idxs)
        cut = int(len(idxs) * (1 - test_ratio))
        cut = min(max(cut, 1), len(idxs) - 1)
        train_idx.extend(idxs[:cut])
        test_idx.extend(idxs[cut:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return x_train, x_test, y_train, y_test


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[List[int]]:
    idx = {c: i for i, c in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1
    return cm


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def classification_report(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    cm = confusion_matrix(y_true, y_pred, labels)
    report: Dict[str, Dict[str, float]] = {}
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(cm[i])
        report[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return report


def _svg_header(width: int, height: int, title: str) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-size="18" font-family="Arial">{title}</text>',
    ]


def save_confusion_matrix_svg(cm: List[List[int]], labels: List[str], path: Path, acc: float) -> None:
    size = 420
    cell = 90
    left = 110
    top = 80
    maxv = max(max(row) for row in cm) or 1

    def color(v: int) -> str:
        t = v / maxv
        blue = int(255 - 140 * t)
        red = int(245 - 220 * t)
        green = int(248 - 180 * t)
        return f"rgb({red},{green},{blue})"

    lines = _svg_header(size + 120, size + 120, f"Confusion Matrix (count, acc={acc:.4f})")
    lines.extend([
        f'<text x="20" y="{top + 1.5*cell}" transform="rotate(-90 20 {top + 1.5*cell})" font-size="14">True label</text>',
        f'<text x="{left + 1.1*cell}" y="{top + 3.6*cell}" font-size="14">Predicted label</text>',
    ])

    for i, label in enumerate(labels):
        lines.append(f'<text x="{left + i*cell + 28}" y="{top - 10}" font-size="14">{label}</text>')
        lines.append(f'<text x="{left - 55}" y="{top + i*cell + 52}" font-size="14">{label}</text>')

    for r in range(len(labels)):
        for c in range(len(labels)):
            x = left + c * cell
            y = top + r * cell
            v = cm[r][c]
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color(v)}" stroke="#888"/>')
            lines.append(f'<text x="{x + 35}" y="{y + 52}" font-size="20" font-family="Arial">{v}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_matrix_normalized_svg(cm: List[List[int]], labels: List[str], path: Path) -> None:
    norm = []
    for row in cm:
        s = sum(row) or 1
        norm.append([v / s for v in row])

    size = 420
    cell = 90
    left = 110
    top = 80

    def color(v: float) -> str:
        blue = int(255 - 140 * v)
        red = int(245 - 220 * v)
        green = int(248 - 180 * v)
        return f"rgb({red},{green},{blue})"

    lines = _svg_header(size + 120, size + 120, "Confusion Matrix (row-normalized)")
    lines.extend([
        f'<text x="20" y="{top + 1.5*cell}" transform="rotate(-90 20 {top + 1.5*cell})" font-size="14">True label</text>',
        f'<text x="{left + 1.1*cell}" y="{top + 3.6*cell}" font-size="14">Predicted label</text>',
    ])

    for i, label in enumerate(labels):
        lines.append(f'<text x="{left + i*cell + 28}" y="{top - 10}" font-size="14">{label}</text>')
        lines.append(f'<text x="{left - 55}" y="{top + i*cell + 52}" font-size="14">{label}</text>')

    for r in range(len(labels)):
        for c in range(len(labels)):
            x = left + c * cell
            y = top + r * cell
            v = norm[r][c]
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color(v)}" stroke="#888"/>')
            lines.append(f'<text x="{x + 22}" y="{y + 52}" font-size="18" font-family="Arial">{v*100:.1f}%</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_class_distribution_svg(y_all: List[str], labels: List[str], path: Path) -> None:
    counts = {l: y_all.count(l) for l in labels}
    w, h = 620, 420
    left, top = 80, 70
    chart_w, chart_h = 500, 280
    bar_w = 110
    maxv = max(counts.values()) or 1
    colors = {"walk": "#4C78A8", "run": "#F58518", "jump": "#54A24B"}

    lines = _svg_header(w, h, "Class Distribution")
    lines.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#333"/>')

    for i, lab in enumerate(labels):
        val = counts[lab]
        bh = (val / maxv) * (chart_h - 20)
        x = left + 60 + i * 150
        y = top + chart_h - bh
        lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="{colors.get(lab, "#4C78A8")}"/>')
        lines.append(f'<text x="{x+35}" y="{top+chart_h+25}" font-size="14">{lab}</text>')
        lines.append(f'<text x="{x+40}" y="{y-8}" font-size="14">{val}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _polyline_points(sig: List[float], x0: int, y0: int, w: int, h: int) -> str:
    smin, smax = min(sig), max(sig)
    rng = (smax - smin) or 1.0
    pts = []
    n = len(sig)
    for i, v in enumerate(sig):
        x = x0 + (i / max(n - 1, 1)) * w
        y = y0 + h - ((v - smin) / rng) * h
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def save_signal_examples_svg(samples: List[SampleWindow], labels: List[str], path: Path) -> None:
    picked: Dict[str, List[float]] = {}
    for lab in labels:
        for s in samples:
            if s.label == lab:
                picked[lab] = s.signal
                break

    w, h = 860, 520
    panel_w, panel_h = 240, 160
    lines = _svg_header(w, h, "Signal Examples by Class")
    colors = {"walk": "#4C78A8", "run": "#F58518", "jump": "#54A24B"}

    for i, lab in enumerate(labels):
        x = 40 + i * 270
        y = 90
        lines.append(f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" fill="white" stroke="#999"/>')
        lines.append(f'<text x="{x+10}" y="{y-10}" font-size="14">{lab}</text>')
        sig = picked.get(lab, [0.0, 0.0])
        pts = _polyline_points(sig, x + 8, y + 8, panel_w - 16, panel_h - 16)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{colors.get(lab, "#333")}" stroke-width="1.6"/>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_feature_separation_svg(x_test: List[List[float]], y_test: List[str], path: Path) -> None:
    # feature idx: rms=2, zcr=5
    rms_vals = [row[2] for row in x_test]
    zcr_vals = [row[5] for row in x_test]
    rx0, rx1 = min(rms_vals), max(rms_vals)
    zx0, zx1 = min(zcr_vals), max(zcr_vals)
    rx_rng = (rx1 - rx0) or 1.0
    zx_rng = (zx1 - zx0) or 1.0

    w, h = 700, 500
    left, top, cw, ch = 80, 70, 540, 360
    colors = {"walk": "#4C78A8", "run": "#F58518", "jump": "#54A24B"}

    lines = _svg_header(w, h, "Feature Scatter (RMS vs ZCR)")
    lines.append(f'<line x1="{left}" y1="{top+ch}" x2="{left+cw}" y2="{top+ch}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ch}" stroke="#333"/>')
    lines.append(f'<text x="{left+cw/2-20}" y="{top+ch+35}" font-size="14">RMS</text>')
    lines.append(f'<text x="20" y="{top+ch/2}" transform="rotate(-90 20 {top+ch/2})" font-size="14">ZCR</text>')

    for f, lab in zip(x_test, y_test):
        xr = (f[2] - rx0) / rx_rng
        zr = (f[5] - zx0) / zx_rng
        x = left + xr * cw
        y = top + ch - zr * ch
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{colors.get(lab, "#666")}" fill-opacity="0.7"/>')

    ly = 455
    lx = 120
    for i, lab in enumerate(LABELS):
        x = lx + i * 150
        lines.append(f'<rect x="{x}" y="{ly}" width="14" height="14" fill="{colors[lab]}"/>')
        lines.append(f'<text x="{x+20}" y="{ly+12}" font-size="13">{lab}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def compute_feature_importance_fscore(x: List[List[float]], y: List[str], labels: List[str]) -> List[float]:
    # 以单特征 ANOVA F-score 作为可解释的重要性近似
    n = len(x)
    if n == 0:
        return []
    m = len(x[0])
    scores: List[float] = []
    for j in range(m):
        col = [row[j] for row in x]
        global_mean = statistics.fmean(col)

        ss_between = 0.0
        ss_within = 0.0
        k = 0
        for lab in labels:
            vals = [x[i][j] for i, yy in enumerate(y) if yy == lab]
            if not vals:
                continue
            k += 1
            mean_c = statistics.fmean(vals)
            ss_between += len(vals) * (mean_c - global_mean) ** 2
            ss_within += sum((v - mean_c) ** 2 for v in vals)

        df_between = max(k - 1, 1)
        df_within = max(n - k, 1)
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within if ss_within > 0 else 1e-12
        scores.append(ms_between / ms_within)
    return scores


def save_feature_importance_svg(feature_scores: List[float], path: Path) -> None:
    feature_names = [
        "mean",
        "std",
        "rms",
        "mav",
        "peak_to_peak",
        "zcr",
        "crest_factor",
        "q25",
        "q50",
        "q75",
        "spec_centroid",
        "spec_entropy",
    ]
    pairs = list(zip(feature_names, feature_scores))
    pairs.sort(key=lambda t: t[1], reverse=True)
    top = pairs[:10]

    w, h = 760, 500
    left, top_y = 200, 60
    bar_h, gap = 30, 10
    maxv = max((v for _, v in top), default=1.0) or 1.0

    lines = _svg_header(w, h, "Top-10 Feature Importance (ANOVA F-score)")
    for i, (name, val) in enumerate(top):
        y = top_y + i * (bar_h + gap)
        bw = 460 * (val / maxv)
        lines.append(f'<rect x="{left}" y="{y}" width="{bw}" height="{bar_h}" fill="#4C78A8"/>')
        lines.append(f'<text x="{left-10}" y="{y+20}" font-size="13" text-anchor="end">{name}</text>')
        lines.append(f'<text x="{left+bw+8}" y="{y+20}" font-size="12">{val:.2f}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_probability_hist_svg(
    y_pred: List[str], probas: List[Dict[str, float]], labels: List[str], path: Path
) -> None:
    # 按预测类别，统计“最大后验概率”分布（5个bin）
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    counts: Dict[str, List[int]] = {lab: [0] * 5 for lab in labels}
    for pred, pb in zip(y_pred, probas):
        conf = max(pb.values())
        for i in range(5):
            if bins[i] <= conf < bins[i + 1]:
                counts[pred][i] += 1
                break

    w, h = 760, 480
    left, top = 70, 80
    cw, ch = 620, 300
    colors = {"walk": "#4C78A8", "run": "#F58518", "jump": "#54A24B"}
    maxv = max((max(v) for v in counts.values()), default=1) or 1

    lines = _svg_header(w, h, "Prediction Confidence Distribution")
    lines.append(f'<line x1="{left}" y1="{top+ch}" x2="{left+cw}" y2="{top+ch}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ch}" stroke="#333"/>')

    group_w = cw / 5
    bar_w = 30
    for b in range(5):
        gx = left + b * group_w + 20
        label = f"{bins[b]:.1f}-{bins[b+1]:.1f}" if b < 4 else "0.9-1.0"
        lines.append(f'<text x="{gx+25}" y="{top+ch+25}" font-size="12">{label}</text>')
        for i, lab in enumerate(labels):
            v = counts[lab][b]
            bh = (v / maxv) * (ch - 20)
            x = gx + i * (bar_w + 8)
            y = top + ch - bh
            lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="{colors[lab]}"/>')

    ly = 410
    lx = 130
    for i, lab in enumerate(labels):
        x = lx + i * 150
        lines.append(f'<rect x="{x}" y="{ly}" width="14" height="14" fill="{colors[lab]}"/>')
        lines.append(f'<text x="{x+20}" y="{ly+12}" font-size="13">{lab}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="TENG 跑跳走分类")
    ap.add_argument("--data-dir", type=Path, default=None, help="真实数据目录，含 walk/run/jump 子目录")
    ap.add_argument("--signal-column", type=str, default="voltage")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic-n", type=int, default=400)
    args = ap.parse_args()

    if args.data_dir and args.data_dir.exists():
        samples = load_real_dataset(args.data_dir, args.signal_column)
        if len(samples) < 90:
            raise RuntimeError(f"真实数据窗口数不足：{len(samples)}，建议至少 90。")
        print(f"[INFO] 使用真实数据窗口：{len(samples)}")
    else:
        samples = generate_synthetic_dataset(args.synthetic_n, seed=args.seed)
        print(f"[INFO] 使用合成数据窗口：{len(samples)}")

    x = [extract_features(s.signal) for s in samples]
    y = [s.label for s in samples]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.25, seed=args.seed)
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    probas = model.predict_proba(x_test)

    cm = confusion_matrix(y_test, y_pred, LABELS)
    acc = accuracy(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, LABELS)
    feature_scores = compute_feature_importance_fscore(x_train, y_train, LABELS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = args.out_dir / "confusion_matrix.svg"
    cm_norm_path = args.out_dir / "confusion_matrix_normalized.svg"
    dist_path = args.out_dir / "class_distribution.svg"
    sig_path = args.out_dir / "signal_examples.svg"
    scatter_path = args.out_dir / "feature_scatter_rms_zcr.svg"
    feat_imp_path = args.out_dir / "feature_importance.svg"
    prob_hist_path = args.out_dir / "prediction_confidence_hist.svg"

    save_confusion_matrix_svg(cm, LABELS, cm_path, acc)
    save_confusion_matrix_normalized_svg(cm, LABELS, cm_norm_path)
    save_class_distribution_svg(y, LABELS, dist_path)
    save_signal_examples_svg(samples, LABELS, sig_path)
    save_feature_separation_svg(x_test, y_test, scatter_path)
    save_feature_importance_svg(feature_scores, feat_imp_path)
    save_probability_hist_svg(y_pred, probas, LABELS, prob_hist_path)

    metrics = {
        "accuracy": acc,
        "labels": LABELS,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "n_samples": len(samples),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "outputs": {
            "confusion_matrix_svg": str(cm_path),
            "confusion_matrix_normalized_svg": str(cm_norm_path),
            "class_distribution_svg": str(dist_path),
            "signal_examples_svg": str(sig_path),
            "feature_scatter_rms_zcr_svg": str(scatter_path),
            "feature_importance_svg": str(feat_imp_path),
            "prediction_confidence_hist_svg": str(prob_hist_path),
        },
    }
    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Result ===")
    print(f"accuracy: {acc:.4f}")
    for lab in LABELS:
        row = cls_report[lab]
        print(
            f"{lab}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}, support={int(row['support'])}"
        )
    for k, v in metrics["outputs"].items():
        print(f"{k}: {v}")
    print(f"metrics: {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
