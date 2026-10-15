# -*- coding: utf-8 -*-
"""
规则引擎基类与通用规则模板。

统一接口：
    rule.fit(train_df)                 -> 在训练折内估计统计量
    rule.predict_sr(case, item)        -> 返回 predicted_SR（float）
    rule.predict_frame(df, item)       -> 批量向量化预测

所有统计量只在 fit 阶段接触训练折数据，predict 阶段仅使用判前信息。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6
FULL_SR_TOL = 1e-4          # 判定 SR == 1 的容差
SR_MIN, SR_MAX = 0.0, 1.0


def _clip(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(min(SR_MAX, max(SR_MIN, x)))


def _get_claim(case: Any, item: str) -> float:
    """从案件级行或分项级行中取出该项目的请求金额（判前信息）"""
    if isinstance(case, dict):
        v = case.get("claim_amount", None)
        if v is None:
            v = case.get(f"{item}_claim", np.nan)
    else:
        if "claim_amount" in case.index:
            v = case["claim_amount"]
        else:
            v = case.get(f"{item}_claim", np.nan)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = np.nan
    return 0.0 if np.isnan(v) else v


class RuleBase:
    """规则基类：统一 fit / predict_sr / predict_frame 接口"""

    name = "rule_base"
    label = "基础规则"

    def __init__(self, item: str, **params: Any):
        self.item = item
        self.params = params
        self.fitted = False
        self.fit_info: dict = {}

    # ---- 接口 ----
    def fit(self, train_df: pd.DataFrame, **_: Any) -> "RuleBase":
        raise NotImplementedError

    def predict_sr(self, case: Any, item: str | None = None) -> float:
        raise NotImplementedError

    def predict_frame(self, df: pd.DataFrame, item: str | None = None) -> np.ndarray:
        """默认逐行实现；子类可重写为向量化版本"""
        it = item or self.item
        return np.array([self.predict_sr(row, it) for _, row in df.iterrows()], dtype=float)

    # ---- 工具 ----
    def _check(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name}[{self.item}] 尚未 fit，禁止预测")

    def describe(self) -> str:
        return f"{self.name}({self.item}) params={self.params} fit_info={self.fit_info}"


class CapRule(RuleBase):
    """
    合理额度锚定规则（cap rule）。

    司法逻辑：该项目存在法定标准 / 凭据据实 / 配置标准，法院以「合理额度 R」为锚，
    claim 未超出 R 时全额支持，超出部分按 R/claim 压缩。

    R 的估计（仅用训练折）：
        R = quantile( claim | SR == 1 , q )
    若训练折中全额支持样本不足（< min_full），退化为
        R = quantile(claim, q) * median(SR)

    predict:  SR = min(1, R / claim)，claim <= 0 时返回 0。
    """

    name = "cap_rule"
    label = "合理额度锚定规则"

    def __init__(self, item: str, q: float = 0.90, min_full: int = 5):
        super().__init__(item, q=q, min_full=min_full)
        self.R = np.nan
        self.median_sr = np.nan

    def fit(self, train_df: pd.DataFrame, **_: Any) -> "CapRule":
        sr = train_df["support_rate_label"].astype(float)
        cl = train_df["claim_amount"].astype(float)
        q = float(self.params.get("q", 0.90))
        min_full = int(self.params.get("min_full", 5))

        full = sr >= (1.0 - FULL_SR_TOL)
        n_full = int(full.sum())
        self.median_sr = float(sr.median()) if len(sr) else 0.0

        if n_full >= min_full and cl[full].gt(0).any():
            R = float(np.quantile(cl[full][cl[full] > 0], q))
            basis = f"quantile(SR=1 样本的 claim, q={q})"
        else:
            R = float(np.quantile(cl[cl > 0], q)) * self.median_sr if cl.gt(0).any() else 0.0
            basis = f"退化：quantile(claim, q={q}) × median(SR)（全额支持样本不足 {min_full}）"

        self.R = max(R, 0.0)
        self.fit_info = {"R": round(self.R, 2), "n_train": int(len(sr)),
                         "n_full_support": n_full, "median_SR": round(self.median_sr, 6),
                         "R_basis": basis}
        self.fitted = True
        return self

    def predict_sr(self, case: Any, item: str | None = None) -> float:
        self._check()
        it = item or self.item
        claim = _get_claim(case, it)
        if claim <= 0:
            return 0.0
        if self.R <= 0:
            return _clip(self.median_sr)
        return _clip(self.R / claim)

    def predict_frame(self, df: pd.DataFrame, item: str | None = None) -> np.ndarray:
        self._check()
        it = item or self.item
        cl = df["claim_amount"].astype(float).to_numpy() \
            if "claim_amount" in df.columns else df[f"{it}_claim"].astype(float).to_numpy()
        out = np.zeros(len(cl), dtype=float)
        pos = cl > 0
        if self.R <= 0:
            out[pos] = _clip(self.median_sr)
        else:
            out[pos] = np.clip(self.R / cl[pos], SR_MIN, SR_MAX)
        return out


class StratifiedMedianRule(RuleBase):
    """
    分层中位数规则：按指定判前字段（默认伤残等级）分组，取训练折内各组的 SR 中位数。
    组内样本不足时回落到全局中位数；字段缺失时同样回落。
    """

    name = "stratified_median_rule"
    label = "分层中位数规则"

    def __init__(self, item: str, key: str = "disability_level_clean", min_group: int = 5):
        super().__init__(item, key=key, min_group=min_group)
        self.table: dict = {}
        self.group_n: dict = {}
        self.global_median = np.nan

    def fit(self, train_df: pd.DataFrame, **_: Any) -> "StratifiedMedianRule":
        key = str(self.params.get("key", "disability_level_clean"))
        min_group = int(self.params.get("min_group", 5))
        sr = train_df["support_rate_label"].astype(float)
        self.global_median = float(sr.median()) if len(sr) else 0.0

        if key in train_df.columns:
            k = train_df[key]
            grp = pd.DataFrame({"k": k, "sr": sr}).dropna(subset=["k"]).groupby("k")["sr"]
            for kk, s in grp:
                if len(s) >= min_group:
                    self.table[float(kk)] = float(s.median())
                    self.group_n[float(kk)] = int(len(s))
        self.fit_info = {"key": key, "n_levels": len(self.table),
                         "global_median": round(self.global_median, 6),
                         "table": {str(k): round(v, 6) for k, v in sorted(self.table.items())}}
        self.fitted = True
        return self

    def predict_sr(self, case: Any, item: str | None = None) -> float:
        self._check()
        key = str(self.params.get("key", "disability_level_clean"))
        v = case[key] if (hasattr(case, "get") and key in case.index) else np.nan
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        if np.isnan(v) or v not in self.table:
            return _clip(self.global_median)
        return _clip(self.table[v])

    def predict_frame(self, df: pd.DataFrame, item: str | None = None) -> np.ndarray:
        key = str(self.params.get("key", "disability_level_clean"))
        if key not in df.columns:
            return np.full(len(df), _clip(self.global_median), dtype=float)
        k = pd.to_numeric(df[key], errors="coerce")
        return k.map(self.table).fillna(_clip(self.global_median)).astype(float).to_numpy()


class QuantilePriorRule(RuleBase):
    """
    分位先验规则（C 类 P2 fallback）：存在规则约束但无完整计算公式时，
    以请求金额分位档 × 训练折该档 SR 中位数作为预测，并受 [0, 1] 约束。
    """

    name = "quantile_prior_rule"
    label = "分位先验规则（规则约束 + 统计先验）"

    def __init__(self, item: str, n_bins: int = 3):
        super().__init__(item, n_bins=n_bins)
        self.cuts: list = []
        self.bin_medians: list = []
        self.global_median = np.nan

    def fit(self, train_df: pd.DataFrame, **_: Any) -> "QuantilePriorRule":
        n_bins = int(self.params.get("n_bins", 3))
        sr = train_df["support_rate_label"].astype(float)
        cl = train_df["claim_amount"].astype(float)
        self.global_median = float(sr.median()) if len(sr) else 0.0

        pos = cl > 0
        if pos.sum() >= n_bins * 2:
            qs = np.quantile(cl[pos], np.linspace(0, 1, n_bins + 1)[1:-1])
            self.cuts = [float(x) for x in np.unique(qs)]
            bins = np.digitize(cl[pos], self.cuts)
            tmp = pd.DataFrame({"b": bins, "sr": sr[pos]})
            agg = tmp.groupby("b")["sr"].median()
            self.bin_medians = [float(agg.get(i, self.global_median)) for i in range(len(self.cuts) + 1)]
        else:
            self.cuts, self.bin_medians = [], [self.global_median]
        self.fit_info = {"n_bins": n_bins, "cuts": [round(c, 2) for c in self.cuts],
                         "bin_medians": [round(v, 6) for v in self.bin_medians],
                         "global_median": round(self.global_median, 6)}
        self.fitted = True
        return self

    def predict_sr(self, case: Any, item: str | None = None) -> float:
        self._check()
        it = item or self.item
        claim = _get_claim(case, it)
        if claim <= 0:
            return 0.0
        idx = int(np.digitize(claim, self.cuts)) if self.cuts else 0
        idx = min(idx, len(self.bin_medians) - 1)
        return _clip(self.bin_medians[idx])

    def predict_frame(self, df: pd.DataFrame, item: str | None = None) -> np.ndarray:
        it = item or self.item
        cl = df["claim_amount"].astype(float).to_numpy() \
            if "claim_amount" in df.columns else df[f"{it}_claim"].astype(float).to_numpy()
        out = np.zeros(len(cl), dtype=float)
        pos = cl > 0
        if pos.any():
            idx = np.digitize(cl[pos], self.cuts) if self.cuts else np.zeros(pos.sum(), dtype=int)
            idx = np.clip(idx, 0, len(self.bin_medians) - 1)
            out[pos] = np.clip(np.asarray(self.bin_medians, dtype=float)[idx], SR_MIN, SR_MAX)
        return out


class GlobalMedianRule(RuleBase):
    """全局中位数规则（C 类 P3 fallback / 缺省兜底）"""

    name = "global_median_rule"
    label = "全局中位数规则"

    def __init__(self, item: str):
        super().__init__(item)
        self.median = np.nan

    def fit(self, train_df: pd.DataFrame, **_: Any) -> "GlobalMedianRule":
        sr = train_df["support_rate_label"].astype(float)
        self.median = float(sr.median()) if len(sr) else 0.0
        self.fit_info = {"median": round(self.median, 6), "n_train": int(len(sr))}
        self.fitted = True
        return self

    def predict_sr(self, case: Any, item: str | None = None) -> float:
        self._check()
        return _clip(self.median)

    def predict_frame(self, df: pd.DataFrame, item: str | None = None) -> np.ndarray:
        return np.full(len(df), _clip(self.median), dtype=float)
