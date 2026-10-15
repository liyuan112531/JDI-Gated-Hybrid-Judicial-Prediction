# -*- coding: utf-8 -*-
"""
伤残相关规则：disability_comp（残疾赔偿金）、assistive_device（残疾辅助器具费）

司法逻辑：
  - 残疾赔偿金 = 受诉法院所在地上一年度城镇居民人均可支配收入 × 赔偿年限 × 伤残系数。
    伤残等级（1 级最重、10 级最轻）直接决定伤残系数，故以伤残等级分层取训练折 SR 中位数。
  - 残疾辅助器具费 = 配置机构意见 + 普通适用器具标准，属于有标准但需合理性判断，
    采用合理额度锚定（CapRule）。
"""
from __future__ import annotations

from .rule_base import CapRule, StratifiedMedianRule


class DisabilityCompRule(StratifiedMedianRule):
    """
    残疾赔偿金：按 disability_level_clean 分层取训练折 SR 中位数。

    分层依据：伤残等级 → 伤残赔偿系数（1 级 100% … 10 级 10%），
    等级越高（伤情越重）法定额度越高、请求越接近法定值，支持率系统性更高。
    """

    name = "disability_comp_rule"
    label = "残疾赔偿金伤残等级分层规则"

    def __init__(self, item: str = "disability_comp",
                 key: str = "disability_level_clean", min_group: int = 5):
        super().__init__(item, key=key, min_group=min_group)


class AssistiveDeviceRule(CapRule):
    """残疾辅助器具费：配置机构意见 + 普通适用标准 → 合理额度锚定（q=0.75）"""

    name = "assistive_device_rule"
    label = "残疾辅助器具费规则"

    def __init__(self, item: str = "assistive_device", q: float = 0.75, min_full: int = 5):
        super().__init__(item, q=q, min_full=min_full)


DISABILITY_RULES = {
    "disability_comp": DisabilityCompRule,
    "assistive_device": AssistiveDeviceRule,
}
