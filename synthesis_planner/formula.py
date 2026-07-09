"""Formula parsing and target-family heuristics."""

from __future__ import annotations

from collections import Counter
import re

TOKEN_RE = re.compile(r"([A-Z][a-z]?|\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?|\(|\)|\[|\]|\{|\}|[·.])")
IGNORED_TOKENS = {"x", "y", "z", "δ", "Δ", "-", "+"}


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a simple inorganic formula into element counts.

    This supports nested parentheses and hydrate separators, which is enough for
    common inorganic formulas such as ``La(NO3)3·6H2O`` or ``BaTiO3``.
    """

    cleaned = formula.strip()
    for token in IGNORED_TOKENS:
        cleaned = cleaned.replace(token, "")

    parts = re.split(r"[·.]", cleaned)
    total = Counter()
    for part in parts:
        if not part:
            continue
        multiplier = 1.0
        leading = re.match(r"^(\d+(?:\.\d+)?)(.*)$", part)
        if leading:
            multiplier = _as_number(leading.group(1))
            part = leading.group(2)
        total.update(_parse_segment(part, multiplier))
    if not total:
        raise ValueError(f"Could not parse formula: {formula}")
    return dict(total)


def _parse_segment(segment: str, multiplier: float) -> Counter:
    tokens = TOKEN_RE.findall(segment)
    if not tokens:
        raise ValueError(f"Could not parse segment: {segment}")

    stack = [Counter()]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("(", "[", "{"):
            stack.append(Counter())
        elif token in (")", "]", "}"):
            if len(stack) == 1:
                raise ValueError(f"Unbalanced formula segment: {segment}")
            group = stack.pop()
            group_multiplier = 1.0
            if i + 1 < len(tokens) and _is_number(tokens[i + 1]):
                group_multiplier = _as_number(tokens[i + 1])
                i += 1
            for element, amount in group.items():
                stack[-1][element] += amount * group_multiplier
        elif _is_number(token):
            raise ValueError(f"Unexpected standalone number in {segment}")
        elif token in (".", "·"):
            pass
        else:
            amount = 1.0
            if i + 1 < len(tokens) and _is_number(tokens[i + 1]):
                amount = _as_number(tokens[i + 1])
                i += 1
            stack[-1][token] += amount
        i += 1

    if len(stack) != 1:
        raise ValueError(f"Unbalanced formula segment: {segment}")

    result = Counter()
    for element, amount in stack[0].items():
        result[element] += amount * multiplier
    return result


def _is_number(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", token))


def _as_number(token: str) -> float:
    if "/" in token:
        numerator, denominator = token.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return float(numerator)
        return float(numerator) / denominator_value
    return float(token)


def normalized_composition(formula: str) -> dict[str, float]:
    counts = parse_formula(formula)
    total = sum(counts.values())
    return {element: amount / total for element, amount in counts.items()}


def required_target_elements(formula: str) -> set[str]:
    counts = parse_formula(formula)
    return {element for element in counts if element not in {"O", "H"}}


def infer_target_class(formula: str) -> str:
    elements = set(parse_formula(formula))
    if "O" in elements and "P" in elements:
        return "phosphate"
    if "S" in elements and "O" not in elements:
        return "sulfide"
    if "N" in elements and "O" not in elements:
        return "nitride"
    if elements & {"F", "Cl", "Br", "I"}:
        return "halide"
    if "O" in elements:
        return "oxide"
    return "other"
