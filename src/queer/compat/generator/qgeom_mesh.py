"""Why two QGEOM adaptive meshes differ: the first divergence and the tied decisions behind it.

QUEER records every decision of its integration that rounding can flip
(``qgeom.integration.decision_ties``, :meth:`queer.observables.qgeom.Integrator.decision_record`):
the occupation pass's seeds, each must-split round, each refinement step and the second
stage's A3 maximum, with the decisions within their rounding band (tied), per rule the
number and an additive digest of the leaves split and the vertex ids their splits created.
:func:`attribute` locates the first point where two runs decided differently and says
whether the runs' records show tied decisions there that produce the difference:

- two records: the seeds, then each round in order. A round agrees when its splits per
  rule (count and digest), its stop and its A3 maximum agree; the meshes before the first
  round that does not are then the same. The difference there is identified only if the
  leaves split outside the decisions tied in both runs are the same (digests of disjoint
  sets add, so the tied leaves are taken out of each digest) and each tied decision
  decided differently could have gone the other run's way in both records; a refinement
  stop decided differently must be tied in both records; a stage whose error estimate
  normalises a quantity that vanishes identically (its scale is rounding) orders its
  leaves by rounding noise, which identifies any difference of its steps;
- one record (a driver reference): the seed count, then the first history entry that
  differs (the driver keeps its last 12, aligned to the record's full history). The
  decision round that produced it must show ties that give the difference: a split
  adds 3 leaves and 4 to 8 vertices (:meth:`~queer.observables.qgeom.AdaptiveMesh.split`),
  so at most the tied decisions that may or may not split change the split count
  (|d leaves| / 3), and each tied decision changes at most 8 vertices; a phase that
  ended in one run and not the other needs a tied end. A flip in the other run puts the
  recording run's margin within twice the rounding, inside its band.

A record is first checked against its own run (:func:`record_problems`): its rounds
must chain from the base mesh to the run's vertex count, its history entries must be
the run's, its counts its lists, and, with the vertex keys in creation order, each
round's splits must be the cells whose child centres that round created, each tied
leaf a leaf of the mesh then and its recorded outcome what the keys show. A record that
is not its run's is never a reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

from ...observables.qgeom import OUTCOMES, AdaptiveMesh, key_hashes, seed_id

HISTORY_EXACT = ("phase", "vertices", "leaves", "target")
# The driver keeps the last entries of its history.
HISTORY_KEPT = 12
REMEDY = "run both on one BLAS kernel (the same OPENBLAS_CORETYPE and numpy build) to compare the meshes"
_MOD = 1 << 64


class Attribution(NamedTuple):
    """The first divergence of two meshes: whether rounding at tied decisions explains it, where and why."""

    identified: bool
    point: str
    detail: str


class MeshRun(NamedTuple):
    """One run's mesh documents: its role, ``qgeom.integration`` block, decision record and vertex keys."""

    role: str
    integration: Mapping[str, Any]
    record: Mapping[str, Any] | None
    keys: np.ndarray | None
    unit: int | None


# ------------------------------------------------------------------ helpers
def _sum(keys: Any) -> int:
    keys = np.asarray(keys, np.int64)
    return int(np.sum(key_hashes(keys), dtype=np.uint64)) if keys.size else 0


def _hex(value: Any) -> int:
    try:
        return int(str(value), 16) % _MOD
    except ValueError:
        return -1


def _exact(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(entry.get(key) for key in HISTORY_EXACT)


def _rows(entry: Mapping[str, Any]) -> dict[int, tuple[int, int]]:
    """A round's tied decisions: centre key -> (mask of possible outcomes, the run's outcome)."""
    return {int(row[0]): (int(row[1]), int(row[2])) for row in entry.get("tied") or []}


def _complete(entry: Mapping[str, Any]) -> bool:
    return len(entry.get("tied") or []) == entry.get("tied_count", 0)


def _outcomes(mask: int) -> list[str]:
    return [name for i, name in enumerate(OUTCOMES) if mask >> i & 1]


def lowest_bit(x: np.ndarray) -> np.ndarray:
    return x & -x


def creating_splits(keys: np.ndarray, unit: int) -> np.ndarray:
    """For each vertex key, the centre key of the cell whose split created it as a child centre, else -1.

    A vertex whose two coordinates have the same lowest set bit l is the centre of the cell
    of size 2 l around it, which exists only once its parent (size 4 l) is split; a base
    cell's centre (l = half a base cell) has no parent.
    """
    keys = np.asarray(keys, np.int64)
    x, y = keys // unit, keys % unit
    lx, ly = lowest_bit(x), lowest_bit(y)
    child = (lx == ly) & (lx > 0) & (lx < (1 << (AdaptiveMesh.DEPTH - 1)))
    size = np.where(child, 4 * lx, 1)
    px = x - x % size + size // 2
    py = y - y % size + size // 2
    return np.where(child, px * unit + py, -1)


def child_centres(centre: int, unit: int) -> list[int]:
    """The centre keys of a cell's four children."""
    x, y = divmod(int(centre), unit)
    quarter = (x & -x) // 2
    return [(x + sx * quarter) * unit + (y + sy * quarter) for sx in (-1, 1) for sy in (-1, 1)]


def history_from_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The run's full history, rebuilt from its decision rounds (a must-split round logs its state after
    its splits, a refinement step before them)."""
    entries: dict[int, dict[str, Any]] = {}
    for entry in record.get("rounds") or []:
        index = entry.get("history")
        if not isinstance(index, int):
            continue
        if entry.get("phase") == "must_split":
            entries[index] = {
                "phase": "must_split",
                "vertices": entry["vertices"][1],
                "leaves": entry["leaves"][1],
                "target": None,
            }
        else:
            entries[index] = {
                "phase": "error",
                "vertices": entry["vertices"][0],
                "leaves": entry["leaves"][0],
                "target": entry.get("target"),
            }
    return [entries[i] for i in sorted(entries)]


# --------------------------------------------------------- record checks
MALFORMED = (KeyError, IndexError, TypeError, ValueError, AttributeError)


def record_problems(run: MeshRun) -> list[str]:
    """Where a run's decision record is not its run's integration and mesh (empty when it is); a record of
    an unexpected shape is a problem, never an error."""
    if run.record is None:
        return []
    try:
        return _record_problems(run)
    except MALFORMED as error:
        return [f"the {run.role}'s decision record is malformed: {type(error).__name__}: {error}"]


def _record_problems(run: MeshRun) -> list[str]:
    record, integration = run.record, run.integration
    assert record is not None
    who = f"the {run.role}'s decision record"
    problems: list[str] = []
    rounds = list(record.get("rounds") or [])
    seeds = record.get("seeds")
    if isinstance(seeds, Mapping):
        if seeds.get("count") != integration.get("seeds"):
            problems.append(f"{who} counts {seeds.get('count')} seeds, its run {integration.get('seeds')}")
        if len(seeds.get("tied") or []) != seeds.get("tied_count"):
            problems.append(
                f"{who} lists {len(seeds.get('tied') or [])} tied seeds of {seeds.get('tied_count')}"
            )
    base = integration.get("base_mesh")
    expected_vertices = 2 * base * base if isinstance(base, int) else None
    expected_leaves = base * base if isinstance(base, int) else None
    for index, entry in enumerate(rounds):
        vertices, leaves = entry.get("vertices") or [None, None], entry.get("leaves") or [None, None]
        if expected_vertices is not None and vertices[0] != expected_vertices:
            return problems + [
                f"{who}: round {index} starts at {vertices[0]} vertices, not {expected_vertices}"
            ]
        if expected_leaves is not None and leaves[0] != expected_leaves:
            return problems + [f"{who}: round {index} starts at {leaves[0]} leaves, not {expected_leaves}"]
        splits = sum(int(n) for n in (entry.get("splits") or {}).values())
        if not (isinstance(leaves[1], int) and leaves[1] == leaves[0] + 3 * splits):
            return problems + [
                f"{who}: round {index} splits {splits} leaves but goes {leaves[0]} -> {leaves[1]}"
            ]
        if not (isinstance(vertices[1], int) and 4 * splits <= vertices[1] - vertices[0] <= 8 * splits):
            return problems + [
                f"{who}: round {index} splits {splits} leaves but goes {vertices[0]} -> {vertices[1]} vertices"
            ]
        if entry.get("outside"):
            return problems + [
                f"{who}: round {index} decided {entry['outside']} leaves outside their rules' bounds"
            ]
        if (
            entry.get("phase") in ("must_split", "error")
            and not _complete(entry)
            and not record.get("truncated")
        ):
            return problems + [
                f"{who}: round {index} lists {len(entry.get('tied') or [])} of {entry.get('tied_count')} ties"
            ]
        expected_vertices, expected_leaves = vertices[1], leaves[1]
    if rounds and expected_vertices != integration.get("vertices"):
        problems.append(
            f"{who} ends at {expected_vertices} vertices, its run at {integration.get('vertices')}"
        )
    history = history_from_record(record)
    kept = list(integration.get("history") or [])
    tail = history[-len(kept) :] if kept else []
    if [_exact(entry) for entry in tail] != [_exact(entry) for entry in kept]:
        problems.append(f"{who}'s history is not its run's (last {len(kept)} entries)")
    if run.keys is not None and run.unit and not problems:
        problems += _key_problems(run, rounds, who)
    return problems


def _key_problems(run: MeshRun, rounds: Sequence[Mapping[str, Any]], who: str) -> list[str]:
    """Each round's splits and tied leaves against the vertex keys in creation order."""
    keys = np.asarray(run.keys, np.int64)
    unit = int(run.unit or 0)
    creator = creating_splits(keys, unit)
    position = {int(k): i for i, k in enumerate(keys.tolist())}
    for index, entry in enumerate(rounds):
        start, end = entry["vertices"]
        for rule, span in (entry.get("ranges") or {}).items():
            lo, hi = span
            if not start <= lo <= hi <= end:
                return [f"{who}: round {index} {rule} range {span} outside {entry['vertices']}"]
            parents, counts = np.unique(creator[lo:hi][creator[lo:hi] >= 0], return_counts=True)
            if np.any(counts != 4) or len(parents) != (entry.get("splits") or {}).get(rule, 0):
                return [f"{who}: round {index} {rule} splits are not the cells its vertices show"]
            if f"{_sum(parents):016x}" != (entry.get("digest") or {}).get(rule):
                return [f"{who}: round {index} {rule} digest is not the cells its vertices show"]
        created = creator[start:end]
        if np.count_nonzero(created >= 0) != 4 * sum(int(n) for n in (entry.get("splits") or {}).values()):
            return [f"{who}: round {index} created child centres its splits do not account for"]
        for centre, (mask, outcome) in _rows(entry).items():
            at = position.get(centre)
            children = [position.get(c) for c in child_centres(centre, unit)]
            if at is None or at >= start or any(c is not None and c < start for c in children):
                return [f"{who}: round {index} lists {centre} as tied, which is not a leaf of the mesh then"]
            if not mask >> outcome & 1:
                return [f"{who}: round {index} gives {centre} an outcome outside its possible ones"]
            if outcome:
                lo, hi = (entry.get("ranges") or {}).get(OUTCOMES[outcome], (0, 0))
                split_here = all(c is not None and lo <= c < hi for c in children)
            else:
                split_here = any(c is not None and start <= c < end for c in children)
            if bool(outcome) != split_here:
                return [
                    f"{who}: round {index} records {centre} as {OUTCOMES[outcome]}, which its vertices contradict"
                ]
    return []


# ---------------------------------------------------------- attribution
def attribute(reference: MeshRun, candidate: MeshRun) -> Attribution:
    """The first divergence of the two meshes and whether the runs' tied decisions explain it."""
    try:
        return _attribute(reference, candidate)
    except MALFORMED as error:
        return Attribution(
            False, "unknown", f"a decision record could not be read: {type(error).__name__}: {error}"
        )


def _attribute(reference: MeshRun, candidate: MeshRun) -> Attribution:
    if reference.record is not None and candidate.record is not None:
        return _two_sided(reference, candidate)
    if reference.record is not None or candidate.record is not None:
        recording, other = (reference, candidate) if reference.record is not None else (candidate, reference)
        return _one_sided(recording, other)
    return Attribution(
        False,
        "unknown",
        "neither run records its tied decisions (integration.decision_ties): the difference cannot be attributed",
    )


def _two_sided(a: MeshRun, b: MeshRun) -> Attribution:
    assert a.record is not None and b.record is not None
    seeds = _seed_difference(a.record.get("seeds"), b.record.get("seeds"))
    if seeds is not None:
        return seeds
    rounds_a, rounds_b = list(a.record.get("rounds") or []), list(b.record.get("rounds") or [])
    for index, (x, y) in enumerate(zip(rounds_a, rounds_b)):
        result = _round_difference(index, x, y, (a.role, b.role))
        if result is not None:
            return result
    if len(rounds_a) != len(rounds_b):
        longer = a if len(rounds_a) > len(rounds_b) else b
        return Attribution(
            False,
            f"decision round {min(len(rounds_a), len(rounds_b))}",
            f"the runs decided alike for {min(len(rounds_a), len(rounds_b))} rounds and the {longer.role} went on "
            "after the other had stopped: a stop no tied decision explains",
        )
    return Attribution(
        False,
        "none",
        "every decision of the two records agrees, yet the vertex sets differ: a record is not its run's mesh",
    )


def _seed_difference(sa: Any, sb: Any) -> Attribution | None:
    point = "the occupation pass (seeds)"
    if not isinstance(sa, Mapping) or not isinstance(sb, Mapping):
        if sa == sb:
            return None
        return Attribution(False, point, "one run records its seed decisions and the other does not")
    if sa.get("count") == sb.get("count") and sa.get("digest") == sb.get("digest"):
        if sa.get("mu_seeds") == sb.get("mu_seeds"):
            return None
        # The same seeds, another count inside the mu range: their energies at its ends.
        delta = abs(int(sa.get("mu_seeds") or 0) - int(sb.get("mu_seeds") or 0))
        tied = min(int(sa.get("mu_seeds_tied") or 0), int(sb.get("mu_seeds_tied") or 0))
        detail = (
            f"{sa.get('mu_seeds')} vs {sb.get('mu_seeds')} seed energies inside the mu range, "
            f"{tied} of them within rounding of its ends in both runs"
        )
        return Attribution(delta <= tied, "the occupation pass (controlled mu)", detail)

    def rows(record: Mapping[str, Any]) -> dict[int, int]:
        return {
            seed_id(i, ("gap", "extremum")[kind], b0, b1): taken
            for i, kind, b0, b1, taken in record.get("tied") or []
        }

    ra, rb = rows(sa), rows(sb)
    both = ra.keys() & rb.keys()
    residual_a = (_hex(sa.get("digest")) - _sum([k for k in both if ra[k]])) % _MOD
    residual_b = (_hex(sb.get("digest")) - _sum([k for k in both if rb[k]])) % _MOD
    flipped = [k for k in both if ra[k] != rb[k]]
    counts = f"{sa.get('count')} vs {sb.get('count')} seeds"
    complete = len(sa.get("tied") or []) == sa.get("tied_count") and len(sb.get("tied") or []) == sb.get(
        "tied_count"
    )
    if complete and residual_a == residual_b and flipped:
        return Attribution(
            True,
            point,
            f"{counts}: {len(flipped)} of the {len(both)} seed decisions tied in both runs (a strict local "
            "extremum or a threshold within rounding of the eigenvalues) went the other way, and every other "
            "seed agrees",
        )
    return Attribution(
        False,
        point,
        f"{counts}: the seeds differ outside the {len(both)} seed decisions tied in both runs",
    )


def _round_name(index: int, entry: Mapping[str, Any]) -> str:
    phase, history = entry.get("phase"), entry.get("history")
    if phase == "must_split":
        where = f"history entry {history}" if isinstance(history, int) else "the round that split nothing"
        return f"must-split round {index} ({where})"
    if phase == "error":
        return f"refinement step {index} (history entry {history}, target {entry.get('target')})"
    return f"round {index} ({phase})"


def _same_round(x: Mapping[str, Any], y: Mapping[str, Any]) -> bool:
    if x.get("phase") != y.get("phase"):
        return False
    if x.get("phase") == "restart":
        return x.get("chosen_mu_eV") == y.get("chosen_mu_eV")
    return (
        (x.get("splits") or {}) == (y.get("splits") or {})
        and (x.get("digest") or {}) == (y.get("digest") or {})
        and (x.get("stop") or {}).get("by") == (y.get("stop") or {}).get("by")
    )


def _noise(entry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """A refinement step's rounding-level quantities when they decide it: some leaf's indicator or the
    worst summed error is one of theirs."""
    noise = entry.get("noise")
    if not isinstance(noise, Mapping) or not noise.get("columns"):
        return None
    worst = (noise.get("worst_rounding") or 0.0) >= (noise.get("worst_real") or 0.0)
    return noise if noise.get("leaves") or worst else None


def _round_difference(
    index: int, x: Mapping[str, Any], y: Mapping[str, Any], roles: tuple[str, str]
) -> Attribution | None:
    if _same_round(x, y):
        return None
    point = _round_name(index, x)
    if x.get("phase") != y.get("phase"):
        return Attribution(
            False,
            point,
            f"after identical decisions the {roles[0]} made a {x.get('phase')} round and the "
            f"{roles[1]} a {y.get('phase')} round",
        )
    if x.get("phase") == "restart":
        mu_x, mu_y = x.get("chosen_mu_eV"), y.get("chosen_mu_eV")
        ok = mu_x in (y.get("tied_mu_eV") or []) and mu_y in (x.get("tied_mu_eV") or [])
        detail = (
            f"the second stage controls the A3 maximum at {mu_x} eV in the {roles[0]} and at {mu_y} eV in the "
            f"{roles[1]}"
        )
        if ok:
            return Attribution(
                True,
                "the second stage's A3 maximum",
                f"{detail}: both lie within rounding of the maximum in both runs",
            )
        return Attribution(
            False, "the second stage's A3 maximum", f"{detail}, which the runs' ties do not cover"
        )
    if x.get("phase") == "error":
        by_x, by_y = (x.get("stop") or {}).get("by"), (y.get("stop") or {}).get("by")
        noise_x, noise_y = _noise(x), _noise(y)
        if by_x != by_y:
            stops = (
                f"the {roles[0]} {'stopped (' + str(by_x) + ')' if by_x else 'went on'}, the {roles[1]} "
                + (f"stopped ({by_y})" if by_y else "went on")
            )
            if {by_x, by_y} == {"target", None}:
                if (x.get("stop") or {}).get("tied") and (y.get("stop") or {}).get("tied"):
                    sx = x["stop"]
                    return Attribution(
                        True,
                        point,
                        f"{stops}: worst <= target is tied in both runs (worst {sx.get('worst')}, "
                        f"target {x.get('target')}, band {sx.get('band')})",
                    )
                if noise_x and noise_y:
                    return Attribution(True, point, f"{stops}: {_noise_text(x, noise_x)}")
            return Attribution(False, point, f"{stops}, and that stop is not tied in both runs")
        if noise_x and noise_y:
            return Attribution(
                True, point, _noise_text(x, noise_x) + f"; splits {x.get('splits')} vs {y.get('splits')}"
            )
    return _tie_difference(point, x, y, roles)


def _noise_text(entry: Mapping[str, Any], noise: Mapping[str, Any]) -> str:
    return (
        f"the stage's error estimate normalises {noise.get('columns')} controlled quantities whose scale is rounding "
        "(they vanish identically), so their estimates are rounding noise; they set the indicator of "
        f"{noise.get('leaves')} leaves and the worst summed error is {noise.get('worst_rounding'):.3g} from them, "
        f"{noise.get('worst_real'):.3g} from the others"
    )


def _tie_difference(
    point: str, x: Mapping[str, Any], y: Mapping[str, Any], roles: tuple[str, str]
) -> Attribution:
    if not _complete(x) or not _complete(y):
        return Attribution(False, point, "a record lists only part of its tied decisions there (truncated)")
    tx, ty = _rows(x), _rows(y)
    both = tx.keys() & ty.keys()
    rules = set(x.get("digest") or {}) | set(y.get("digest") or {})
    outside = []
    for rule in sorted(rules):
        code = OUTCOMES.index(rule)
        residual_x = (
            _hex((x.get("digest") or {}).get(rule, "0")) - _sum([k for k in both if tx[k][1] == code])
        ) % _MOD
        residual_y = (
            _hex((y.get("digest") or {}).get(rule, "0")) - _sum([k for k in both if ty[k][1] == code])
        ) % _MOD
        if residual_x != residual_y:
            outside.append(rule)
    flipped = [k for k in both if tx[k][1] != ty[k][1]]
    unreachable = [k for k in flipped if not (tx[k][0] >> ty[k][1] & 1 and ty[k][0] >> tx[k][1] & 1)]
    splits = f"splits {x.get('splits')} vs {y.get('splits')}"
    if outside:
        return Attribution(
            False,
            point,
            f"{splits}: the leaves split by the {', '.join(outside)} rule differ outside the {len(both)} decisions "
            f"tied in both runs ({x.get('tied_count')} and {y.get('tied_count')} tied in each)",
        )
    if not flipped or unreachable:
        return Attribution(False, point, f"{splits}: the tied decisions do not account for the difference")
    pairs: dict[str, int] = {}
    for k in flipped:
        label = f"{OUTCOMES[tx[k][1]]} vs {OUTCOMES[ty[k][1]]}"
        pairs[label] = pairs.get(label, 0) + 1
    rules_text = ", ".join(f"{label} x{n}" for label, n in sorted(pairs.items()))
    by_rule = x.get("tied_by_rule")
    extra = f"; tied per rule {by_rule}" if by_rule else ""
    if x.get("phase") == "error":
        extra = f"; the cut at indicator {(x.get('cut') or {}).get('value')} within {(x.get('cut') or {}).get('band')}"
    return Attribution(
        True,
        point,
        f"{splits}: {len(flipped)} of the {len(both)} decisions tied in both runs went the other way "
        f"({rules_text}), and every other leaf agrees{extra}",
    )


# ------------------------------------------------------ one-sided record
def _one_sided(recording: MeshRun, other: MeshRun) -> Attribution:
    record = recording.record
    assert record is not None
    ir, io = recording.integration, other.integration
    if io.get("seeds") != ir.get("seeds"):
        delta = abs(int(io.get("seeds") or 0) - int(ir.get("seeds") or 0))
        tied = int((record.get("seeds") or {}).get("tied_count") or 0)
        detail = (
            f"{io.get('seeds')} seeds in the {other.role}, {ir.get('seeds')} in the {recording.role}, whose record "
            f"shows {tied} tied seed decisions"
        )
        return Attribution(
            isinstance(io.get("seeds"), int) and delta <= tied, "the occupation pass (seeds)", detail
        )
    full = history_from_record(record)
    kept = list(io.get("history") or [])
    if not kept:
        return Attribution(False, "unknown", f"the {other.role} records no history to locate the difference")
    if len(kept) < HISTORY_KEPT:
        offset: int | None = 0
    else:
        offset = next((i for i, entry in enumerate(full) if _exact(entry) == _exact(kept[0])), None)
    rounds = list(record.get("rounds") or [])
    if offset is None:
        seen = (
            f"the {other.role} keeps its last {len(kept)} history entries, from {kept[0]}, which the "
            f"{recording.role} never reached: the first divergence precedes them"
        )
        must_split = ("base", "seed", "gap", "edge")
        by_rule_r, by_rule_o = ir.get("vertices_by_rule") or {}, io.get("vertices_by_rule") or {}
        steps = [r for r in rounds if r.get("phase") == "error"]
        if (
            all(by_rule_r.get(k) == by_rule_o.get(k) for k in must_split)
            and ir.get("gap_rule_capped") == io.get("gap_rule_capped")
            and steps
            and all(_noise(r) for r in steps)
        ):
            counts = {k: by_rule_r.get(k) for k in must_split}
            return Attribution(
                True,
                "the refinement, before the recorded history",
                f"{seen}; the must-split phase agrees in its vertices per rule {counts}, and every refinement "
                f"step of the {recording.role} is ordered by rounding: {_noise_text(steps[0], _noise(steps[0]) or {})}",
            )
        return Attribution(False, "before the recorded history", seen)
    for j, entry in enumerate(kept):
        e = offset + j
        if e >= len(full) or _exact(full[e]) != _exact(entry):
            return _entry_difference(recording, other, rounds, full, e, entry)
    if len(full) != offset + len(kept):
        return _entry_difference(recording, other, rounds, full, offset + len(kept), None)
    return _fields_difference(recording, other, rounds)


def _producing_round(rounds: Sequence[Mapping[str, Any]], full: Sequence[Mapping[str, Any]], e: int):
    """The recording run's decision round whose decisions produced history entry ``e`` (its last round when
    its history ends before ``e``): a must-split entry's own round, else the round before the entry's."""
    if e < len(full) and full[e]["phase"] == "must_split":
        return next((r for r in rounds if r.get("history") == e), None)
    at = next((i for i, r in enumerate(rounds) if r.get("history") == e), len(rounds))
    earlier = [r for r in rounds[:at] if r.get("phase") != "restart"]
    return earlier[-1] if earlier else None


def _flips(entry: Mapping[str, Any]) -> tuple[int, int]:
    """(decisions that may or may not split a leaf, all tied decisions) of a round."""
    tied = int(entry.get("tied_count") or 0)
    if entry.get("phase") == "error":
        take = entry.get("take") or {}
        spread = (take.get("range") or [0, 0])[1] - (take.get("range") or [0, 0])[0]
        return int(spread), tied
    split_codes = sum(1 << OUTCOMES.index(rule) for rule in ("seed", "gap", "edge"))
    flips = sum(1 for mask, _ in _rows(entry).values() if mask & 1 and mask & split_codes)
    return (flips if _complete(entry) else tied), tied


def _entry_difference(recording, other, rounds, full, e, entry) -> Attribution:
    mine = full[e] if e < len(full) else None
    point = f"history entry {e}"
    seen = f"{other.role} {entry} vs {recording.role} {mine}"
    producer = _producing_round(rounds, full, e)
    if producer is None:
        return Attribution(False, point, f"{seen}: no decision round of the record precedes it")
    name = _round_name(rounds.index(producer), producer)
    noise = _noise(producer)
    if noise:
        return Attribution(True, f"{point}, from {name}", f"{seen}: {_noise_text(producer, noise)}")
    restart = next((r for r in rounds if r.get("phase") == "restart"), None)
    if (
        mine is None
        or entry is None
        or (mine["phase"], mine.get("target")) != (entry.get("phase"), entry.get("target"))
    ):
        # One run ended a phase or a stage where the other went on.
        stop = producer.get("stop") or {}
        if producer.get("phase") == "error" and stop.get("tied"):
            return Attribution(True, f"{point}, from {name}", f"{seen}: its stop worst <= target is tied")
        if producer.get("phase") == "must_split":
            flips, _ = _flips(producer)
            total = sum(int(n) for n in (producer.get("splits") or {}).values())
            if (total == 0 and flips) or (total and flips >= total):
                return Attribution(
                    True,
                    f"{point}, from {name}",
                    f"{seen}: its {flips} tied decisions can end or extend the phase",
                )
        return Attribution(False, f"{point}, from {name}", f"{seen}: a phase ended in one run only, untied")
    if restart is not None and producer is not None and rounds.index(producer) > rounds.index(restart):
        if (restart.get("tied_count") or 0) > 0:
            return Attribution(True, f"{point}", f"{seen}: after the second stage's tied A3 maximum")
    d_leaves = int(entry["leaves"]) - int(mine["leaves"])
    d_vertices = int(entry["vertices"]) - int(mine["vertices"])
    flips, tied = _flips(producer)
    ok = d_leaves % 3 == 0 and abs(d_leaves) // 3 <= flips and abs(d_vertices) <= 8 * flips + 4 * tied
    detail = (
        f"{seen}: {d_leaves:+d} leaves ({d_leaves / 3:+g} splits) and {d_vertices:+d} vertices against "
        f"{flips} tied split decisions and {tied} tied decisions in all of {name}"
    )
    return Attribution(ok, f"{point}, from {name}", detail)


def record_seeds(run: MeshRun) -> Mapping[str, Any] | None:
    seeds = (run.record or {}).get("seeds")
    return seeds if isinstance(seeds, Mapping) else None


def _fields_difference(recording, other, rounds) -> Attribution:
    ir, io = recording.integration, other.integration
    fields = [
        k for k in ("vertices_by_rule", "mu_subset", "gap_rule_capped", "max_level") if ir.get(k) != io.get(k)
    ]
    if not fields:
        return Attribution(False, "none", "the recorded meshes agree in every recorded field")
    point = f"the recorded {', '.join(fields)}"
    if fields == ["mu_subset"]:
        restart = next((r for r in rounds if r.get("phase") == "restart"), None)
        if restart is not None and (restart.get("tied_count") or 0) > 0:
            return Attribution(True, point, "the history agrees; the second stage's A3 maximum is tied")
        if int((record_seeds(recording) or {}).get("mu_seeds_tied") or 0) > 0:
            return Attribution(
                True, point, "the history agrees; seed energies lie within rounding of the mu range's ends"
            )
    if fields == ["vertices_by_rule"]:
        split_rules = [OUTCOMES.index(rule) for rule in ("seed", "gap", "edge")]
        shared = sum(
            1
            for r in rounds
            if r.get("phase") == "must_split"
            for mask, _ in _rows(r).values()
            if sum(mask >> code & 1 for code in split_rules) >= 2
        )
        a, b = ir.get("vertices_by_rule") or {}, io.get("vertices_by_rule") or {}
        delta = max(abs(int(a.get(k, 0)) - int(b.get(k, 0))) for k in set(a) | set(b))
        if shared and delta <= 8 * shared:
            return Attribution(
                True,
                point,
                f"the history agrees; {shared} tied decisions between two split rules move up to "
                f"{8 * shared} vertices between rules (largest difference {delta})",
            )
    return Attribution(
        False, point, f"the history agrees but {', '.join(fields)} differ, which no tie explains"
    )
