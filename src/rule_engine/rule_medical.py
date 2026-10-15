# -*- coding: utf-8 -*-
"""
医疗费类规则：medical（医疗费）、hospital_meal（住院伙食补助费）

司法逻辑：
  - 医疗费：凭医疗费票据、病历、诊断证明据实计算；无票据或非必要治疗部分被削减。
    判前可得的最强锚是「请求金额是否落在合理额度内」。
  - 住院伙食补助费：固定日标准 × 住院天数，属法定标准，几乎按标准全额支持；
    超出合理额度的请求被压缩。

均采用 CapRule（合理额度锚定），R 由训练折内全额支持（SR=1）样本的请求金额分位数估计。
"""
from __future__ import annotations

from .rule_base import CapRule


class MedicalRule(CapRule):
    """医疗费：凭据据实 → 合理额度锚定（R = 训练折 SR=1 样本 claim 的 90 分位）"""

    name = "medical_rule"
    label = "医疗费规则"

    def __init__(self, item: str = "medical", q: float = 0.90, min_full: int = 5):
        super().__init__(item, q=q, min_full=min_full)


class HospitalMealRule(CapRule):
    """住院伙食补助费：固定日标准 × 住院天数 → 合理额度锚定（q=0.90）"""

    name = "hospital_meal_rule"
    label = "住院伙食补助费规则"

    def __init__(self, item: str = "hospital_meal", q: float = 0.90, min_full: int = 5):
        super().__init__(item, q=q, min_full=min_full)


MEDICAL_RULES = {
    "medical": MedicalRule,
    "hospital_meal": HospitalMealRule,
}
