"""Route retrieval and precursor prior construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product

from .formula import normalized_composition, parse_formula, required_target_elements
from .schema import PrecursorRecord, RouteRecord


class RetrievalIndex:
    def __init__(self, routes: list[RouteRecord]):
        self.routes = routes
        self.precursor_usage_by_element = defaultdict(Counter)
        self.class_usage_by_element = defaultdict(Counter)

        for route in routes:
            used = {element for precursor in route.precursors for element in precursor.elements}
            for element in used:
                for precursor in route.precursors:
                    if element in precursor.elements:
                        self.precursor_usage_by_element[element][precursor.formula] += 1
                        self.class_usage_by_element[element][precursor.class_name] += 1

    def retrieve(self, target_formula: str, top_k: int = 12) -> list[tuple[float, RouteRecord]]:
        target_elements = set(parse_formula(target_formula))
        target_class = _target_class_from_routes(self.routes, target_formula)
        target_comp = _safe_normalized_composition(target_formula, tuple(sorted(target_elements)))

        scored = []
        for route in self.routes:
            route_elements = set(route.target_elements)
            overlap = len(target_elements & route_elements)
            union = len(target_elements | route_elements)
            jaccard = overlap / union if union else 0.0
            comp_distance = 0.0
            route_comp = _safe_normalized_composition(route.target_formula, route.target_elements)
            for element in target_elements | route_elements:
                comp_distance += abs(target_comp.get(element, 0.0) - route_comp.get(element, 0.0))
            score = 3.0 * jaccard - comp_distance
            if route.target_formula == target_formula:
                score += 5.0
            if route.target_class == target_class:
                score += 1.0
            if required_target_elements(target_formula) <= {element for precursor in route.precursors for element in precursor.elements}:
                score += 0.5
            scored.append((score, route))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def candidate_precursor_sets(self, target_formula: str, analogs: list[tuple[float, RouteRecord]], max_sets: int = 10) -> list[tuple[float, tuple[PrecursorRecord, ...]]]:
        required = required_target_elements(target_formula)
        candidates: dict[tuple[str, ...], tuple[float, tuple[PrecursorRecord, ...]]] = {}

        for score, route in analogs:
            coverage = {element for precursor in route.precursors for element in precursor.elements}
            if required <= coverage and len(route.precursors) <= 8:
                key = tuple(sorted(precursor.formula for precursor in route.precursors))
                candidates[key] = (score + 1.0, route.precursors)

        per_element = []
        for element in sorted(required):
            options = []
            for formula, count in self.precursor_usage_by_element[element].most_common(5):
                precursor = PrecursorRecord(
                    formula=formula,
                    class_name=_most_common_class(formula, analogs),
                    elements=tuple(sorted(parse_formula(formula))),
                )
                options.append((count, precursor))
            if options:
                per_element.append(options[:3])

        if per_element:
            for combo in product(*per_element):
                score = 0.0
                chosen = {}
                for count, precursor in combo:
                    chosen[precursor.formula] = precursor
                    score += float(count)
                precursors = tuple(sorted(chosen.values(), key=lambda item: item.formula))
                coverage = {element for precursor in precursors for element in precursor.elements}
                if required <= coverage:
                    key = tuple(precursor.formula for precursor in precursors)
                    if key not in candidates:
                        candidates[key] = (score / max(1, len(precursors)), precursors)

        ranked = sorted(candidates.values(), key=lambda item: item[0], reverse=True)
        return ranked[:max_sets]


def _target_class_from_routes(routes: list[RouteRecord], target_formula: str) -> str:
    for route in routes:
        if route.target_formula == target_formula:
            return route.target_class
    from .formula import infer_target_class

    return infer_target_class(target_formula)


def _most_common_class(formula: str, analogs: list[tuple[float, RouteRecord]]) -> str:
    counter = Counter()
    for _, route in analogs:
        for precursor in route.precursors:
            if precursor.formula == formula:
                counter[precursor.class_name] += 1
    return counter.most_common(1)[0][0] if counter else "elemental_or_other"


def _safe_normalized_composition(formula: str, elements: tuple[str, ...]) -> dict[str, float]:
    try:
        return normalized_composition(formula)
    except Exception:
        weight = 1.0 / max(1, len(elements))
        return {element: weight for element in elements}
