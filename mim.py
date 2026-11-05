"""TOOL: bit-packed meet-in-the-middle min-weight kernel (numpy).

Same contract as spectrum.min_weight — exact below the cutoff, a returned
value equal to the cutoff means ">= cutoff" (certified bound) — but built
for LARGE signature sets (qLDPC scale), where the pure-Python weight-5
sweep is hopeless:

  packing     each distinct signature -> detector words (uint64[W]) +
              one observable word; multi-word keys compared as raw bytes
              (void views), which is a valid total order for grouping.
  weight 1/2  vectorized scans / byte-sort run detection.
  weight 3    pair XOR joined against the sorted singles table
              (vectorized searchsorted, one outer Python loop).
  weight 4    the materialized pair table, sorted, run-scanned for
              distinct observables at equal detector keys.
  weight 5    triple = (single i) x (pair table), joined back against the
              sorted pair table; outer loop over i, everything inner
              vectorized; optionally multiprocessed over i (fork).

Soundness of the joins leans on the same minimality argument as the pure
version: any overlapping/repeated combination reduces to a smaller
weight, which was excluded before the current level ran (spectrum.py,
min_weight docstring).

Memory note: weight 4/5 materialize the D^2/2 pair table
(W+1 words/row). BB72 rounds=2 (D ~ 4.6k) ~ 0.6 GB; gross-scale D ~ 20k
~ 11 GB -- sized for the EPYC box, not a laptop. procs > 1 uses fork
(shared read-only tables; explicitly requested so macOS spawn semantics
never copy the table per worker).

Validated against spectrum.min_weight on randomized instances and on the
pinned real cases in tests/test_mim.py; cegis/spectrum consumers opt in
explicitly (kernel="numpy").
"""

from multiprocessing import get_context

import numpy as np


def _pack(vals, words):
    out = np.zeros((len(vals), words), dtype=np.uint64)
    mask = (1 << 64) - 1
    for i, v in enumerate(vals):
        for w in range(words):
            out[i, w] = (v >> (64 * w)) & mask
    return out


def _void(arr):
    """Rows as fixed-size byte records (memcmp total order)."""
    a = np.ascontiguousarray(arr)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def _run_summary(sorted_obs, starts, ends):
    """Per run of equal keys: (first observable, has-two-distinct flag)."""
    first = sorted_obs[starts]
    has_two = np.zeros(len(starts), dtype=bool)
    for k in range(len(starts)):
        if ends[k] - starts[k] > 1:
            seg = sorted_obs[starts[k]:ends[k]]
            has_two[k] = bool((seg != seg[0]).any())
    return first, has_two


def _runs(keys_void):
    """starts/ends of equal-key runs in a SORTED void array."""
    n = len(keys_void)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = keys_void[1:] != keys_void[:-1]
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n)
    return starts, ends


_W5 = {}          # fork-shared state for the weight-5 workers


def _w5_block(i_block):
    dets, obs, pair_det, pair_obs, pv_sorted, order, run_first, \
        run_two, run_of_pos = (
            _W5["dets"], _W5["obs"], _W5["pair_det"], _W5["pair_obs"],
            _W5["pv_sorted"], _W5["order"], _W5["run_first"],
            _W5["run_two"], _W5["run_of_pos"])
    P = len(pair_det)
    tile = 1 << 23            # rows per transient chunk: bounds each
    for i in i_block:         # worker's peak memory (OOM discipline)
        for lo in range(0, P, tile):
            hi = min(P, lo + tile)
            tri_det = pair_det[lo:hi] ^ dets[i]
            tv = _void(tri_det)
            pos = np.searchsorted(pv_sorted, tv)
            ok = pos < len(pv_sorted)
            if not ok.any():
                continue
            hit = np.flatnonzero(ok & (pv_sorted[np.minimum(
                pos, len(pv_sorted) - 1)] == tv))
            if len(hit) == 0:
                continue
            runs = run_of_pos[pos[hit]]
            want = pair_obs[lo + hit] ^ obs[i]     # triple's observable
            found = run_two[runs] | (run_first[runs] != want)
            if found.any():
                return True
    return False


def min_weight_packed(sigs, n_det, cutoff=4, procs=1):
    """Drop-in for spectrum.min_weight (cutoff <= 6)."""
    if cutoff > 6:
        raise ValueError("cutoff > 6 not implemented")
    detmask = (1 << n_det) - 1
    S = sorted(set(sigs))
    D = len(S)
    if D == 0:
        return cutoff
    W = max(1, (n_det + 63) // 64)
    if any((s >> n_det) >= (1 << 64) for s in S):
        raise ValueError("more than 64 observable bits")
    dets = _pack([s & detmask for s in S], W)
    obs = np.array([s >> n_det for s in S], dtype=np.uint64)

    zero_det = ~dets.any(axis=1)
    if cutoff > 1 and bool((zero_det & (obs != 0)).any()):
        return 1

    dv = _void(dets)
    order1 = np.argsort(dv, kind="stable")
    dv_sorted = dv[order1]
    s1, e1 = _runs(dv_sorted)
    if cutoff > 2 and bool((e1 - s1 > 1).any()):
        return 2          # distinct signatures, equal det => distinct obs
    if cutoff <= 3:
        return cutoff

    # weight 3: pair XOR meets a single with a different observable
    obs1_sorted = obs[order1]
    r1_first, r1_two = _run_summary(obs1_sorted, s1, e1)
    run_of_pos1 = np.zeros(len(dv_sorted), dtype=np.int64)
    run_of_pos1[s1] = 1
    run_of_pos1 = np.cumsum(run_of_pos1) - 1
    for i in range(D - 1):
        pd = dets[i + 1:] ^ dets[i]
        pv = _void(pd)
        pos = np.searchsorted(dv_sorted, pv)
        ok = pos < len(dv_sorted)
        if not ok.any():
            continue
        hit = np.flatnonzero(ok & (dv_sorted[np.minimum(
            pos, len(dv_sorted) - 1)] == pv))
        if len(hit) == 0:
            continue
        runs = run_of_pos1[pos[hit]]
        want = obs[i + 1:][hit] ^ obs[i]
        if bool((r1_two[runs] | (r1_first[runs] != want)).any()):
            return 3
    if cutoff <= 4:
        return cutoff

    # pair table (upper triangle), sorted by det key
    ii, jj = np.triu_indices(D, k=1)
    pair_det = dets[ii] ^ dets[jj]
    pair_obs = obs[ii] ^ obs[jj]
    pv = _void(pair_det)
    order2 = np.argsort(pv, kind="stable")
    pv_sorted = pv[order2]
    pobs_sorted = pair_obs[order2]
    s2, e2 = _runs(pv_sorted)
    r2_first, r2_two = _run_summary(pobs_sorted, s2, e2)
    if bool(r2_two.any()):
        return 4
    if cutoff <= 5:
        return cutoff

    # weight 5: triples (single x pair table) joined against pair table
    run_of_pos2 = np.zeros(len(pv_sorted), dtype=np.int64)
    run_of_pos2[s2] = 1
    run_of_pos2 = np.cumsum(run_of_pos2) - 1
    _W5.update(dets=dets, obs=obs, pair_det=pair_det, pair_obs=pair_obs,
               pv_sorted=pv_sorted, order=order2, run_first=r2_first,
               run_two=r2_two, run_of_pos=run_of_pos2)
    idx = list(range(D))
    if procs and procs > 1:
        blocks = [idx[k::procs] for k in range(procs)]
        with get_context("fork").Pool(procs) as pool:
            if any(pool.map(_w5_block, blocks)):
                return 5
    else:
        if _w5_block(idx):
            return 5
    return cutoff
