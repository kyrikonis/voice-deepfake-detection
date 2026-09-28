#!/usr/bin/env python
"""
Verify the raw datasets against their protocol files and write a data report.

For every split this checks:
  - the protocol parses with the expected number of columns and label strings
  - which protocol entries have no audio on disk ("missing") and which audio files have no
    protocol entry ("extra")
  - label counts and attack/system counts
  - audio headers (sample rate, channels, duration) on a class-balanced random sample,
    or on every file with --full
  - no utterance ID appears twice within a split or in more than one split, and which
    speakers are shared between splits

Outputs go to <outputs_root>/data_report/:
  report.md           summary tables (pasted into docs/results.md)
  summary.csv         the files-and-labels table, machine-readable
  <split>_audio.csv   one row per header-checked file; the EDA notebook reads these

The protocol parsers here are deliberately small and private to this script. The reusable
parsers in src/vdd/data are written in Phase 1 (tutor mode), so don't import from here.

Usage (inside WSL, venv active, from the repo root):
    python scripts/verify_data.py              # 1000 files per class per split
    python scripts/verify_data.py --full       # every file (slow: ~430k headers over /mnt/d)

"""

# This file was drafted by Claude to verify the datasets for the project.


import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_SR = 16000
CROP_SAMPLES = 64600  # ~4.04 s, the standard ASVspoof input length (see CLAUDE.md)
LABELS = {"bonafide", "spoof"}


# ---------------------------------------------------------------------------
# Protocol parsers. Each returns a DataFrame with at least:
#   utt_id, speaker, label ("bonafide"/"spoof"), attack, filename
# ---------------------------------------------------------------------------

def read_whitespace_table(path, n_cols):
    """Split each non-empty line on whitespace and assert every row has n_cols fields."""
    with open(path) as f:
        rows = [line.split() for line in f if line.strip()]
    bad = [i for i, r in enumerate(rows) if len(r) != n_cols]
    assert not bad, f"{path}: {len(bad)} rows without {n_cols} columns, first at line {bad[0] + 1}"
    return rows


def parse_la19(path):
    # SPEAKER_ID FILE_NAME - SYSTEM_ID KEY
    rows = read_whitespace_table(path, 5)
    df = pd.DataFrame(rows, columns=["speaker", "utt_id", "unused", "attack", "label"])
    df["filename"] = df["utt_id"] + ".flac"
    return df.drop(columns="unused")


def parse_asv5(path):
    # The README says "five columns" but lists (and the files contain) ten.
    cols = ["speaker", "utt_id", "gender", "codec", "codec_q", "codec_seed",
            "attack_tag", "attack", "label", "tmp"]
    rows = read_whitespace_table(path, len(cols))
    df = pd.DataFrame(rows, columns=cols)
    df["filename"] = df["utt_id"] + ".flac"
    return df


def parse_itw(path):
    # meta.csv columns: file,speaker,label with labels "bona-fide" / "spoof"
    df = pd.read_csv(path)
    assert list(df.columns) == ["file", "speaker", "label"], f"unexpected columns {list(df.columns)}"
    raw_labels = set(df["label"])
    assert raw_labels == {"bona-fide", "spoof"}, f"unexpected ITW labels {raw_labels}"
    df["label"] = df["label"].map({"bona-fide": "bonafide", "spoof": "spoof"})
    df["filename"] = df["file"]
    df["utt_id"] = df["file"].str.removesuffix(".wav")
    df["attack"] = "-"  # ITW has no attack labels
    return df.drop(columns="file")


PARSERS = {"la19": parse_la19, "asv5": parse_asv5, "itw": parse_itw}


def split_specs(paths):
    """(name, dataset, protocol key, audio key, partial download allowed)"""
    specs = [
        ("la19_train", "asvspoof2019_la", "train_protocol", "train_flac", "la19", False),
        ("la19_dev", "asvspoof2019_la", "dev_protocol", "dev_flac", "la19", False),
        ("la19_eval", "asvspoof2019_la", "eval_protocol", "eval_flac", "la19", False),
        ("asv5_dev", "asvspoof5", "dev_protocol", "dev_flac", "asv5", False),
        ("asv5_eval", "asvspoof5", "eval_protocol", "eval_flac", "asv5", True),
        ("itw", "itw", "meta", "audio", "itw", False),
    ]
    root = paths["data_root"]
    for name, ds, proto_key, audio_key, kind, partial_ok in specs:
        yield dict(
            name=name,
            kind=kind,
            protocol=os.path.join(root, paths[ds][proto_key]),
            audio_dir=os.path.join(root, paths[ds][audio_key]),
            partial_ok=partial_ok,
        )


# ---------------------------------------------------------------------------
# Audio headers
# ---------------------------------------------------------------------------

def read_header(path):
    """Read only the file header (fast): no audio is decoded."""
    try:
        info = sf.info(path)
        return dict(samplerate=info.samplerate, channels=info.channels,
                    frames=info.frames, subtype=info.subtype, error="")
    except Exception as e:  # corrupt or unreadable file: record it, don't crash
        return dict(samplerate=0, channels=0, frames=0, subtype="", error=repr(e))


def check_headers(df, audio_dir, workers, desc):
    paths = [os.path.join(audio_dir, f) for f in df["filename"]]
    # Header reads are I/O-bound, so threads are enough (no need for processes).
    with ThreadPoolExecutor(max_workers=workers) as pool:
        infos = list(tqdm(pool.map(read_header, paths), total=len(paths), desc=desc, leave=False))
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(infos)], axis=1)
    out["duration_s"] = out["frames"] / out["samplerate"].where(out["samplerate"] > 0)
    return out


def balanced_sample(df, n_per_class, seed):
    """Up to n files per label, so the rarer bona fide class still gets a useful sample."""
    parts = [g.sample(min(len(g), n_per_class), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def md_table(df):
    """Tiny markdown table writer (avoids needing the optional 'tabulate' package)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=os.path.join(REPO_ROOT, "configs", "paths.yaml"))
    ap.add_argument("--full", action="store_true", help="check every file's header, not a sample")
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    out_dir = os.path.join(paths["outputs_root"], "data_report")
    os.makedirs(out_dir, exist_ok=True)

    problems = []   # anything here makes the script exit non-zero
    summary, audio_rows, attack_tables = [], [], []
    protocols = {}

    for spec in split_specs(paths):
        name = spec["name"]
        print(f"== {name}")
        df = PARSERS[spec["kind"]](spec["protocol"])
        protocols[name] = df

        labels = set(df["label"])
        if not labels <= LABELS:
            problems.append(f"{name}: unexpected labels {labels - LABELS}")
        n_dup = df["utt_id"].duplicated().sum()
        if n_dup:
            problems.append(f"{name}: {n_dup} duplicate utterance IDs in protocol")

        # One directory listing instead of one stat() per file: much faster over /mnt/d.
        ext = os.path.splitext(df["filename"].iloc[0])[1]
        on_disk = {f for f in os.listdir(spec["audio_dir"]) if f.endswith(ext)}
        df["present"] = df["filename"].isin(on_disk)
        n_missing = int((~df["present"]).sum())
        n_extra = len(on_disk - set(df["filename"]))
        if n_missing and not spec["partial_ok"]:
            problems.append(f"{name}: {n_missing} protocol entries have no audio file")

        present = df[df["present"]]
        to_check = present if args.full else balanced_sample(present, args.n_per_class, args.seed)
        audio = check_headers(to_check, spec["audio_dir"], args.workers, name)
        audio.to_csv(os.path.join(out_dir, f"{name}_audio.csv"), index=False)

        n_err = int((audio["error"] != "").sum())
        if n_err:
            problems.append(f"{name}: {n_err} unreadable files (see {name}_audio.csv)")

        counts = present["label"].value_counts()
        summary.append(dict(
            split=name,
            protocol_rows=len(df),
            files_on_disk=len(on_disk),
            usable=len(present),
            missing=n_missing,
            extra=n_extra,
            bonafide=int(counts.get("bonafide", 0)),
            spoof=int(counts.get("spoof", 0)),
            headers_checked=len(audio),
            unreadable=n_err,
        ))

        ok = audio[audio["error"] == ""]
        for label, g in ok.groupby("label"):
            audio_rows.append(dict(
                split=name,
                label=label,
                n=len(g),
                sample_rates=", ".join(f"{sr}:{c}" for sr, c in g["samplerate"].value_counts().items()),
                channels=", ".join(f"{ch}:{c}" for ch, c in g["channels"].value_counts().items()),
                dur_min=round(g["duration_s"].min(), 2),
                dur_median=round(g["duration_s"].median(), 2),
                dur_mean=round(g["duration_s"].mean(), 2),
                dur_max=round(g["duration_s"].max(), 2),
                # Share that needs repeat-padding at 16 kHz: frames scaled to 16 kHz first.
                pct_shorter_than_crop=round(100 * (g["duration_s"] * TARGET_SR < CROP_SAMPLES).mean(), 1),
            ))

        attack_tables.append((name, present.groupby(["attack", "label"]).size().rename("n").reset_index()))

    # --- Leakage checks across splits ---------------------------------------------------
    leak_rows = []
    names = list(protocols)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_ids = len(set(protocols[a]["utt_id"]) & set(protocols[b]["utt_id"]))
            if shared_ids:
                problems.append(f"utterance IDs shared between {a} and {b}: {shared_ids}")
            # Speaker overlap is reported, not treated as an error: it's a property of the
            # corpus design (2019 LA speakers are disjoint across train/dev/eval), not a bug.
            shared_spk = len(set(protocols[a]["speaker"]) & set(protocols[b]["speaker"]))
            leak_rows.append(dict(split_a=a, split_b=b, shared_utt_ids=shared_ids, shared_speakers=shared_spk))

    # --- Report ----------------------------------------------------------------------
    mode = "all files" if args.full else f"class-balanced sample of up to {args.n_per_class} per class (seed {args.seed})"
    parts = [
        "# Data report (generated by scripts/verify_data.py)",
        "## Files and labels",
        md_table(pd.DataFrame(summary)),
        "*missing* = protocol rows with no audio; *extra* = audio with no protocol row "
        "(ignored by the loaders). Label counts are over usable files.",
        f"## Audio properties ({mode})",
        md_table(pd.DataFrame(audio_rows)),
        f"`pct_shorter_than_crop` = share of clips shorter than {CROP_SAMPLES} samples at 16 kHz, "
        "which get repeat-padded.",
        "## Overlap between splits",
        md_table(pd.DataFrame(leak_rows)),
        "## Attack / system counts",
    ]
    for name, t in attack_tables:
        wide = t.pivot(index="attack", columns="label", values="n").fillna(0).astype(int).reset_index()
        parts += [f"### {name}", md_table(wide)]
    parts += ["## Problems", "\n".join(f"- {p}" for p in problems) or "None."]

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    report = "\n\n".join(parts) + "\n"
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print(pd.DataFrame(audio_rows).to_string(index=False))
    print()
    print(pd.DataFrame(leak_rows).to_string(index=False))
    print(f"\nReport written to {report_path}")
    if problems:
        print("\nPROBLEMS:\n" + "\n".join(f"  - {p}" for p in problems))
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
