# -*- coding: utf-8 -*-
"""
规则引擎包：rule + data-driven 混合框架中的「规则部分」。

模块：
  rule_base       规则基类与通用模板（CapRule / StratifiedMedianRule / QuantilePriorRule / GlobalMedianRule）
  rule_medical    医疗费类规则（medical / hospital_meal）
  rule_disability 伤残相关规则（disability_comp / assistive_device）
  rule_generic    appraisal 规则、C 类 fallback 注册表、规则工厂

统一接口：
  rule.fit(train_df)           仅用训练折估计统计量
  rule.predict_sr(case, item)  返回 predicted_SR ∈ [0, 1]
"""
from .rule_base import (CapRule, GlobalMedianRule, QuantilePriorRule,
                        RuleBase, StratifiedMedianRule)
from .rule_medical import HospitalMealRule, MedicalRule
from .rule_disability import AssistiveDeviceRule, DisabilityCompRule
from .rule_generic import (A_CLASS_RULES, C_CLASS_FALLBACK, AppraisalRule,
                           RULE_CANDIDATES, build_candidates, build_rule)

__all__ = [
    "RuleBase", "CapRule", "StratifiedMedianRule", "QuantilePriorRule", "GlobalMedianRule",
    "MedicalRule", "HospitalMealRule", "DisabilityCompRule", "AssistiveDeviceRule",
    "AppraisalRule", "A_CLASS_RULES", "C_CLASS_FALLBACK", "build_rule",
    "RULE_CANDIDATES", "build_candidates",
]
