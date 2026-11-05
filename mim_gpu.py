"""TOOL: GPU weight-5 sweep for the packed min-weight kernel (CuPy).

Same contract as mim.min_weight_packed / spectrum.min_weight (exact
below the cutoff; == cutoff means ">= cutoff"). Weights 1-4 are cheap
and reuse the CPU kernel's logic; the O(D^3-shaped) weight-5 sweep runs
on the GPU:

  - multi-word detector keys are folded to 64-bit hashes on-device
    (splitmix-style per-word mixing);
  - the sorted pair-hash table is joined against (single i) x (pair
    table) XORs, one fused pass per i, all on-device;
  - every hash hit is re-verified EXACTLY (full-width keys + the
    observable condition) before weight 5 is reported, so a collision
    can only cost time, never soundness.

Falls back with ImportError if CuPy/CUDA is unavailable. Validated
against mim.min_weight_packed on randomized instances and the pinned
BB72 winner (6/6). Historical validation is summarized in
docs/REPRODUCIBILITY.md; GPU execution requires a suitable CUDA host.
"""

import numpy as np

import cupy as cp

_M1 = np.uint64(0xff51afd7ed558ccd)
_M2 = np.uint64(0xc4ceb9fe1a85ec53)


def _mix(h):
    h = h ^ (h >> cp.uint64(33))
    h = h * cp.uint64(_M1)
    h = h ^ (h >> cp.uint64(33))
    h = h * cp.uint64(_M2)
    return h ^ (h >> cp.uint64(33))


_SALT = 0x9e3779b97f4a7c15


def _hash_rows(words):
    """(N, W) uint64 -> (N,) uint64 order-dependent row hash."""
    salts = cp.arange(words.shape[1], dtype=cp.uint64) * cp.uint64(_SALT)
    h = cp.zeros(words.shape[0], dtype=cp.uint64)
    for w in range(words.shape[1]):
        h = _mix(h ^ words[:, w] ^ salts[w])
    return h


def weight5_gpu(dets_np, obs_np, batch_rows=1 << 26):
    """True iff some 5 distinct signatures XOR to (det=0, obs!=0),
    given that weights 1-4 were already excluded (minimality argument:
    overlapping combinations reduce and were ruled out upstream).
    dets_np: (D, W) uint64; obs_np: (D,) uint64.

    Structure mirrors the CPU kernel: the sorted pair table carries
    per-run summaries (first observable, any-two-distinct flag), so the
    no-witness case -- every winner -- answers each probe vectorized
    with NO fallback scans. Runs are hash-runs; any run summary that
    would report weight 5 is re-verified exactly (full-width keys) so a
    64-bit collision can cost time, never a wrong certified answer."""
    D, W = dets_np.shape
    dets = cp.asarray(dets_np)
    obs = cp.asarray(obs_np)

    ii, jj = np.triu_indices(D, k=1)
    ii = cp.asarray(ii.astype(np.int32))
    jj = cp.asarray(jj.astype(np.int32))
    pair_det = dets[ii] ^ dets[jj]                    # (P, W)
    pair_obs = obs[ii] ^ obs[jj]
    del ii, jj
    P = pair_det.shape[0]

    ph = _hash_rows(pair_det)
    order = cp.argsort(ph)
    ph_sorted = ph[order]
    pobs_sorted = pair_obs[order]

    # hash-run boundaries + per-run summaries (prefix-sum reduction:
    # no ufunc.at / scatter needed)
    new = cp.ones(P, dtype=cp.bool_)
    new[1:] = ph_sorted[1:] != ph_sorted[:-1]
    run_id = cp.cumsum(new) - 1                       # (P,) int64
    starts = cp.nonzero(new)[0]
    ends = cp.concatenate([starts[1:], cp.asarray([P])])
    run_first = pobs_sorted[starts]
    diff = (pobs_sorted != run_first[run_id]).astype(cp.int64)
    prefix = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(diff)])
    run_two = (prefix[ends] - prefix[starts]) > 0
    run_of_pos = run_id

    for i in range(D):
        di = dets[i]
        oi = obs[i]
        for lo in range(0, P, batch_rows):
            hi = min(P, lo + batch_rows)
            th = _hash_rows(pair_det[lo:hi] ^ di)
            pos = cp.searchsorted(ph_sorted, th)
            pos = cp.minimum(pos, P - 1)
            ok = ph_sorted[pos] == th
            if not bool(ok.any()):
                continue
            hit = cp.nonzero(ok)[0]
            runs = run_of_pos[pos[hit]]
            want = pair_obs[lo + hit] ^ oi
            cand = run_two[runs] | (run_first[runs] != want)
            if not bool(cand.any()):
                continue
            # candidate weight-5: verify exactly (full-width key match
            # AND a genuinely differing observable in the equal-key
            # run). Candidates are hash-verified positives or rare
            # collisions; a genuine witness makes the first exact check
            # return True, so this loop is short in practice and
            # complete by construction.
            for t in hit[cand].tolist():
                key = pair_det[lo + t] ^ di
                run = cp.nonzero((pair_det == key[None, :]).all(axis=1))[0]
                if run.size == 0:
                    continue          # pure hash collision
                w = pair_obs[lo + t] ^ oi
                if bool((pair_obs[run] != w).any()):
                    return True
    return False


def min_weight_gpu(sigs, n_det, cutoff=4):
    """Drop-in: CPU kernel for weights 1-4, GPU for the weight-5 sweep."""
    from mim import _pack, min_weight_packed
    if cutoff <= 5:
        return min_weight_packed(sigs, n_det, cutoff)
    base = min_weight_packed(sigs, n_det, 5)
    if base < 5:
        return base
    detmask = (1 << n_det) - 1
    S = sorted(set(sigs))
    W = max(1, (n_det + 63) // 64)
    dets = _pack([s & detmask for s in S], W)
    obs = np.array([s >> n_det for s in S], dtype=np.uint64)
    return 5 if weight5_gpu(dets, obs) else cutoff
