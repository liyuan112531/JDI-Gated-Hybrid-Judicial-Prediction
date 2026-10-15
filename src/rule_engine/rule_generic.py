# -*- coding: utf-8 -*-
"""
通用规则：appraisal（鉴定费）与各类 fallback 规则的注册与工厂。

统一接口：predict_sr(case, item) -> predicted_SR
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from .rule_base import (CapRule, GlobalMedianRule, QuantilePriorRule,
                        StratifiedMedianRule)
from .rule_medical import MEDICAL_RULES, HospitalMealRule, MedicalRule
from .rule_disability import DISABILITY_RULES, AssistiveDeviceRule, DisabilityCompRule


class AppraisalRule(CapRule):
    """鉴定费：凭鉴定费票据据实计算 → 合理额度锚定（q=0.95，支持率极高）"""

    name = "appraisal_rule"
    label = "鉴定费规则"

    def __init__(self, item: str = "appraisal", q: float = 0.95, min_full: int = 5):
        super().__init__(item, q=q, min_full=min_full)


# ---------------------------------------------------------------------------
# A 类（Rule-based）规则注册表
# ---------------------------------------------------------------------------
A_CLASS_RULES: Dict[str, Callable] = {
    **{k: v for k, v in MEDICAL_RULES.items()},
    **{k: v for k, v in DISABILITY_RULES.items()},
    "appraisal": AppraisalRule,
}

# ---------------------------------------------------------------------------
# C 类（Small-sample / Low-support-risk）Rule-informed fallback 注册表
# 优先级 1：存在明确法律计算公式 → 规则计算（CapRule 额度锚定）
# 优先级 2：存在规则约束但无完整公式 → 规则约束 + 训练集 SR 统计先验（QuantilePriorRule）
# 优先级 3：无规则 → 训练集 SR 中位数（GlobalMedianRule）
# ---------------------------------------------------------------------------
C_CLASS_FALLBACK: Dict[str, Dict] = {
    "death_comp": {
        "priority": "P1",
        "rule_class": CapRule,
        "params": {"q": 0.50, "min_full": 3},
        "formula": "SR = min(1, R / claim)，R = 训练折内 SR=1 样本 claim 的中位数（近似法定额度）",
        "legal_basis": "死亡赔偿金 = 人均可支配收入 × 法定年限（法定公式）",
        "reason": "存在明确法定计算公式，按规则计算；请求未超出法定额度时全额支持，超出部分按额度压缩",
    },
    "funeral": {
        "priority": "P1",
        "rule_class": CapRule,
        "params": {"q": 0.50, "min_full": 3},
        "formula": "SR = min(1, R / claim)，R = 训练折内 SR=1 样本 claim 的中位数（近似法定额度）",
        "legal_basis": "丧葬费 = 职工月平均工资 × 6 个月（法定标准）",
        "reason": "存在明确法定标准，按规则计算",
    },
    "accommodation": {
        "priority": "P2",
        "rule_class": QuantilePriorRule,
        "params": {"n_bins": 3},
        "formula": "SR = 训练折内同请求金额分位档的 SR 中位数（clip 到 [0,1]）",
        "legal_basis": "住宿费：凭据 + 合理性酌定（有约束，无完整公式）",
        "reason": "存在规则约束（需凭据且合理）但无完整计算公式，采用规则约束 + 训练集统计先验",
    },
    "future_medical": {
        "priority": "P2",
        "rule_class": QuantilePriorRule,
        "params": {"n_bins": 3},
        "formula": "SR = 训练折内同请求金额分位档的 SR 中位数（clip 到 [0,1]）",
        "legal_basis": "后续医疗费：需鉴定结论或待实际发生后另行主张（有约束，无完整公式）",
        "reason": "存在规则约束（需鉴定或待实际发生）但无完整计算公式，采用规则约束 + 训练集统计先验",
    },
}


def build_rule(item: str) -> object:
    """按项目构建规则对象（A 类规则 或 C 类 fallback）"""
    if item in A_CLASS_RULES:
        return A_CLASS_RULES[item](item)
    if item in C_CLASS_FALLBACK:
        cfg = C_CLASS_FALLBACK[item]
        return cfg["rule_class"](item, **cfg.get("params", {}))
    # 缺省（B 类不适用规则）：全局中位数兜底，仅用于系统集成时的占位
    return GlobalMedianRule(item)


# ---------------------------------------------------------------------------
# 规则变体候选集：用于「训练折内的规则标定（rule selection）」
# 第 1 个候选为法律先验规则（legal prior），其余为数据标定备选。
# 选择过程只在训练折内进行（内部交叉验证），不接触测试折，也不是模型训练。
# ---------------------------------------------------------------------------
_CAP_GRID = [("cap_q50", 0.50), ("cap_q75", 0.75), ("cap_q90", 0.90), ("cap_q95", 0.95)]

RULE_CANDIDATES: Dict[str, list] = {
    "medical": [("medical_rule", MedicalRule, {}),
                *[(n, CapRule, {"q": q}) for n, q in _CAP_GRID],
                ("global_median", GlobalMedianRule, {}),
                ("quantile_prior3", QuantilePriorRule, {"n_bins": 3})],
    "hospital_meal": [("hospital_meal_rule", HospitalMealRule, {}),
                      *[(n, CapRule, {"q": q}) for n, q in _CAP_GRID],
                      ("global_median", GlobalMedianRule, {}),
                      ("quantile_prior3", QuantilePriorRule, {"n_bins": 3})],
    "appraisal": [("appraisal_rule", AppraisalRule, {}),
                  *[(n, CapRule, {"q": q}) for n, q in _CAP_GRID],
                  ("global_median", GlobalMedianRule, {}),
                  ("quantile_prior3", QuantilePriorRule, {"n_bins": 3})],
    "assistive_device": [("assistive_device_rule", AssistiveDeviceRule, {}),
                         *[(n, CapRule, {"q": q}) for n, q in _CAP_GRID],
                         ("global_median", GlobalMedianRule, {}),
                         ("quantile_prior3", QuantilePriorRule, {"n_bins": 3})],
    "disability_comp": [("disability_comp_rule", DisabilityCompRule, {}),
                        ("stratified_claim", StratifiedMedianRule,
                         {"key": "disability_level_clean"}),
                        *[(n, CapRule, {"q": q}) for n, q in _CAP_GRID],
                        ("global_median", GlobalMedianRule, {}),
                        ("quantile_prior3", QuantilePriorRule, {"n_bins": 3})],
    "death_comp": [("death_comp_rule", CapRule, {"q": 0.50, "min_full": 3}),
                   *[(n, CapRule, {"q": q, "min_full": 3}) for n, q in _CAP_GRID],
                   ("global_median", GlobalMedianRule, {}),
                   ("quantile_prior2", QuantilePriorRule, {"n_bins": 2})],
    "funeral": [("funeral_rule", CapRule, {"q": 0.50, "min_full": 3}),
                *[(n, CapRule, {"q": q, "min_full": 3}) for n, q in _CAP_GRID],
                ("global_median", GlobalMedianRule, {}),
                ("quantile_prior2", QuantilePriorRule, {"n_bins": 2})],
    "accommodation": [("accommodation_rule", QuantilePriorRule, {"n_bins": 3}),
                      ("global_median", GlobalMedianRule, {}),
                      ("quantile_prior2", QuantilePriorRule, {"n_bins": 2}),
                      *[(n, CapRule, {"q": q, "min_full": 3}) for n, q in _CAP_GRID]],
    "future_medical": [("future_medical_rule", QuantilePriorRule, {"n_bins": 3}),
                       ("global_median", GlobalMedianRule, {}),
                       ("quantile_prior2", QuantilePriorRule, {"n_bins": 2}),
                       *[(n, CapRule, {"q": q, "min_full": 3}) for n, q in _CAP_GRID]],
}


def build_candidates(item: str) -> list:
    """返回 [(name, rule_instance_factory, params), ...]"""
    out = []
    for name, cls, params in RULE_CANDIDATES.get(item, []):
        out.append((name, cls, params))
    return out


__all__ = [
    "AppraisalRule", "A_CLASS_RULES", "C_CLASS_FALLBACK", "build_rule",
    "CapRule", "GlobalMedianRule", "QuantilePriorRule", "StratifiedMedianRule",
    "RULE_CANDIDATES", "build_candidates",
]
