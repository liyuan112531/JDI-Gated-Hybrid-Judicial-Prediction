# -*- coding: utf-8 -*-
"""
test_reproducibility — 审稿人 3 秒快速核验
==========================================

不重新训练任何模型，仅对随仓库发布的冻结基准产物做一致性断言：
  T1  数据规模：dataset_anonymized.csv = 2,380 行 x 41 列，case_id 唯一；
  T2  防泄漏检查：results/stage1/stage1_leakage_check.json 全部 PASS（4/4）；
  T3  OOF 完整性：3 系统 x 3 repeats x 2,380 案件 = 21,420 行，
      sr_pred / sr_true 均在 [0, 1]，每个 repeat 覆盖全部案件；
  T4  OOF 核心指标：gated_gamma_1 pooled MAE = 0.1670（±0.0005），
      且优于 REF_Full_ML 与 gated_gamma_0（论文表 5 结论方向）；
  T5  冻结基准表格与论文命名一致（Gated Hybrid γ=1 主模型）。

运行方式：
    python tests/test_reproducibility.py          # 直接运行
    pytest tests/test_reproducibility.py -q       # 或 pytest 运行
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "dataset_anonymized.csv"
S1 = BASE / "results" / "stage1"
S10 = BASE / "results" / "stage10_ultimate_sota"
S12 = BASE / "results" / "stage12_paper_updates"

RESULTS = []  # (name, ok, detail)


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))


# --- T1 数据规模 ---
df = pd.read_csv(DATA, encoding="utf-8-sig")
check("T1a 数据行数 = 2380", len(df) == 2380, f"n={len(df)}")
check("T1b 数据列数 = 41", df.shape[1] == 41, f"cols={df.shape[1]}")
check("T1c case_id 唯一", df["case_id"].is_unique)
check("T1d case_id 覆盖 C0000-C2379",
      df["case_id"].iloc[0] == "C0000" and df["case_id"].iloc[-1] == "C2379")

# --- T2 防泄漏检查 4/4 PASS ---
leak = json.loads((S1 / "stage1_leakage_check.json").read_text(encoding="utf-8"))
items = leak.get("checks", leak)
pairs = items.items() if isinstance(items, dict) else enumerate(items)
stat = [(k, (v.get("status") if isinstance(v, dict) else v)) for k, v in pairs]
n_pass = sum(1 for _, s in stat if s in ("PASS", True))
check("T2 泄漏检查全部 PASS（4/4）", len(stat) == 4 and n_pass == 4,
      f"{n_pass}/{len(stat)}")

# --- T3/T4 案件级 OOF ---
oof = pd.read_parquet(S10 / "stage10_oof_case_predictions.parquet")
check("T3a OOF 行数 = 3 系统 x 7140", len(oof) == 3 * 7140, f"n={len(oof)}")
check("T3b 三系统齐全",
      set(oof["system"].unique()) == {"REF_Full_ML", "gated_gamma_0", "gated_gamma_1"})
check("T3c 预测/真值均在 [0,1]",
      bool(oof["sr_pred"].between(0, 1).all() and oof["sr_true"].between(0, 1).all()))
n_per_rep = oof.groupby(["system", "repeat"])["case_id"].nunique()
check("T3d 每个 repeat 覆盖全部 2380 案件", bool((n_per_rep == 2380).all()))


def _metrics(sub):
    err = sub["sr_pred"].to_numpy(float) - sub["sr_true"].to_numpy(float)
    return {"MAE": float(abs(err).mean()), "RMSE": float(((err ** 2).mean()) ** 0.5)}


m_full = _metrics(oof[oof["system"] == "REF_Full_ML"])
m_soft = _metrics(oof[oof["system"] == "gated_gamma_0"])
m_gate = _metrics(oof[oof["system"] == "gated_gamma_1"])
check("T4a Gated Hybrid pooled MAE ≈ 0.1670",
      abs(m_gate["MAE"] - 0.1670) < 5e-4, f"MAE={m_gate['MAE']:.6f}")
check("T4b Gated Hybrid pooled RMSE ≈ 0.2357",
      abs(m_gate["RMSE"] - 0.2357) < 5e-4, f"RMSE={m_gate['RMSE']:.6f}")
check("T4c 优于 Full ML（MAE）", m_gate["MAE"] < m_full["MAE"],
      f"{m_gate['MAE']:.6f} < {m_full['MAE']:.6f}")
check("T4d 优于 Soft Hybrid γ=0（MAE）", m_gate["MAE"] < m_soft["MAE"],
      f"{m_gate['MAE']:.6f} < {m_soft['MAE']:.6f}")

# --- T5 冻结表格命名与关键数值 ---
t5 = pd.read_csv(S12 / "table5_improvement_path.csv", encoding="utf-8-sig")
gate_row = t5[t5.iloc[:, 0].astype(str).str.contains("Gated Hybrid", na=False)]
check("T5a 表5 终行为 Gated Hybrid 主模型",
      len(gate_row) == 1 and t5.index[-1] == gate_row.index[0])
check("T5b 表5 主模型 MAE 与冻结值一致",
      abs(float(gate_row["MAE"].iloc[0]) - 0.16700315960639506) < 1e-12)
t9 = pd.read_csv(S12 / "table9_temporal_validation_2022to2023.csv", encoding="utf-8-sig")
check("T5c 表9 含 Gated_Hybrid_γ=1 且无遗留代号",
      t9["System"].astype(str).str.contains("Gated_Hybrid_γ=1").any()
      and not t9["System"].astype(str).str.contains("Ultimate").any())
t10 = pd.read_csv(S12 / "table10_subgroup_error_analysis.csv", encoding="utf-8-sig")
first = t10.iloc[0]
check("T5d 表10 首行为基准行 n=7,140 (100.00%)",
      str(first["分组"]).startswith("全部袋外样本")
      and str(first["n"]) == "7,140 (100.00%)"
      and abs(float(first["MAE"]) - 0.1670) < 5e-5)

# --- 汇总 ---
n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print("\n" + "=" * 60)
print(f"共 {len(RESULTS)} 项断言：{len(RESULTS) - n_fail} PASS / {n_fail} FAIL")
print("=" * 60)
if __name__ == "__main__":
    sys.exit(1 if n_fail else 0)


def test_frozen_reference_outputs():
    """pytest 入口：任何断言失败即测试失败。"""
    failed = [name for name, ok, _ in RESULTS if not ok]
    assert not failed, f"核验未通过: {failed}"
