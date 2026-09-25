"""
Fault-label mapping for CWRU and Paderborn.
"""

import re

CWRU_PREFIX_TO_FAULT = {
    "normal": "Healthy",
    "b": "Ball",
    "ir": "InnerRace",
    "or": "OuterRace",
}

_CWRU_PATTERN = re.compile(
    r"^(?P<prefix>normal|b|ir|or)(?P<severity>\d{3})?(?:@6)?_(?P<load>\d)$",
    re.IGNORECASE,
)


def cwru_label_from_filename(label_str):
    m = _CWRU_PATTERN.match(label_str)
    if not m:
        return None
    prefix = m.group("prefix").lower()
    fault_type = CWRU_PREFIX_TO_FAULT[prefix]
    severity = m.group("severity")
    load = int(m.group("load"))
    return fault_type, severity, load


CWRU_CLASSES = ["Healthy", "Ball", "InnerRace", "OuterRace"]

PADERBORN_BEARING_MAP = {
    "K001": "Healthy", "K002": "Healthy", "K003": "Healthy",
    "K004": "Healthy", "K005": "Healthy", "K006": "Healthy",
    "KA01": "OuterRace", "KA03": "OuterRace", "KA05": "OuterRace",
    "KA06": "OuterRace", "KA07": "OuterRace", "KA08": "OuterRace", "KA09": "OuterRace",
    "KI01": "InnerRace", "KI03": "InnerRace", "KI05": "InnerRace",
    "KI07": "InnerRace", "KI08": "InnerRace",
    "KA04": "OuterRace", "KA15": "OuterRace", "KA16": "OuterRace",
    "KA22": "OuterRace", "KA30": "OuterRace",
    "KI04": "InnerRace", "KI14": "InnerRace", "KI16": "InnerRace",
    "KI17": "InnerRace", "KI18": "InnerRace", "KI21": "InnerRace",
    "KB23": None, "KB24": None, "KB27": None,
}

PADERBORN_CLASSES = ["Healthy", "InnerRace", "OuterRace"]

_PADERBORN_CODE_PATTERN = re.compile(r"(K[AIB]?\d{2,3})")


def paderborn_label_from_filename(label_str):
    m = _PADERBORN_CODE_PATTERN.search(label_str)
    if not m:
        return None
    code = m.group(1)
    return PADERBORN_BEARING_MAP.get(code, None)