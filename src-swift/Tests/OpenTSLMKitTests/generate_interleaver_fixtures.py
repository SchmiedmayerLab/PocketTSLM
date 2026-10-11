#!/usr/bin/env python3
"""Generate Python↔Swift parity fixtures for SoftPromptInterleaver.padAndInterleaveBatch.

The interleaver is pure tensor assembly (no weights): per sample it trims each segment to
its valid length (mask.sum(), right-padding assumed), lays them out as
[pre, (ts_text_i, ts_emb_i)…, post], concatenates, then zero-pads each sample to the batch
max. Values are irrelevant to the logic, so we use deterministic random embeddings with a
realistic ECG-ish structure: a 12-"lead" sample and a shorter 4-segment sample (to exercise
padding), with intra-segment padding (to exercise the per-segment trim).

Run:  python3 src-swift/Tests/OpenTSLMKitTests/generate_interleaver_fixtures.py
"""

import os
import numpy as np
from safetensors.numpy import save_file

FIXTURES = os.path.dirname(os.path.abspath(__file__)) + "/Fixtures"
H = 8
rng = np.random.default_rng(0)


def seg(total, valid):
    """Random [total, H] embeddings + a right-padded mask with `valid` ones."""
    emb = rng.standard_normal((total, H)).astype(np.float32)
    mask = np.zeros(total, dtype=np.float32)
    mask[:valid] = 1.0
    return emb, mask


def valid_len(mask):
    return int(mask.sum())


def interleave(samples):
    seqs, masks = [], []
    for s in samples:
        se, sm = [], []

        def app(emb_mask):
            emb, mask = emb_mask
            L = valid_len(mask)
            se.append(emb[:L])
            sm.append(np.ones(L, dtype=np.float32))

        app(s["pre"])
        for tt, te in zip(s["tstext"], s["tsemb"]):
            app(tt)
            se.append(te)
            sm.append(np.ones(te.shape[0], dtype=np.float32))
        app(s["post"])
        seqs.append(np.concatenate(se, 0))
        masks.append(np.concatenate(sm, 0))

    max_len = max(x.shape[0] for x in seqs)
    pe, pm = [], []
    for emb, mask in zip(seqs, masks):
        pad = max_len - emb.shape[0]
        if pad > 0:
            emb = np.concatenate([emb, np.zeros((pad, H), np.float32)], 0)
            mask = np.concatenate([mask, np.zeros(pad, np.float32)], 0)
        pe.append(emb)
        pm.append(mask)
    return np.stack(pe, 0), np.stack(pm, 0)


def make_sample(k, pre_lv, tt_lv, te_n, post_lv):
    return {
        "pre": seg(pre_lv + 1, pre_lv),                       # 1 trailing pad
        "tstext": [seg(tt_lv + 1, tt_lv) for _ in range(k)],   # 1 trailing pad each
        "tsemb": [rng.standard_normal((te_n, H)).astype(np.float32) for _ in range(k)],
        "post": seg(post_lv + 1, post_lv),
    }


def main():
    os.makedirs(FIXTURES, exist_ok=True)
    samples = [
        make_sample(k=12, pre_lv=5, tt_lv=2, te_n=4, post_lv=3),  # 12-lead, longer
        make_sample(k=4, pre_lv=3, tt_lv=1, te_n=4, post_lv=2),   # shorter → gets padded
    ]
    out_emb, out_mask = interleave(samples)

    flat = {
        "ts_counts": np.array([len(s["tstext"]) for s in samples], dtype=np.int32),
        "out_embeds": out_emb,
        "out_mask": out_mask,
    }
    for si, s in enumerate(samples):
        flat[f"s{si}_pre_emb"], flat[f"s{si}_pre_mask"] = s["pre"]
        flat[f"s{si}_post_emb"], flat[f"s{si}_post_mask"] = s["post"]
        for i, (tt, te) in enumerate(zip(s["tstext"], s["tsemb"])):
            flat[f"s{si}_tt{i}_emb"], flat[f"s{si}_tt{i}_mask"] = tt
            flat[f"s{si}_te{i}"] = te

    save_file(flat, f"{FIXTURES}/interleaver_io.safetensors")
    print(f"samples={len(samples)} ts_counts={flat['ts_counts'].tolist()} "
          f"H={H} out={out_emb.shape} -> {FIXTURES}/interleaver_io.safetensors")


if __name__ == "__main__":
    main()
