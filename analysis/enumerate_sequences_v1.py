#!/usr/bin/env python3
"""Enumerate a constant-length sequence space, optionally modulo reversal."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def sequence_count(alphabet_size, length, reversal):
    oriented = alphabet_size**length
    palindromes = alphabet_size**((length + 1) // 2)
    return (oriented + palindromes) // 2 if reversal else oriented


def enumerate_sequences(alphabet, length, reversal):
    for residues in itertools.product(sorted(alphabet), repeat=length):
        sequence = "".join(residues)
        if not reversal or sequence <= sequence[::-1]:
            yield sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amino-acids", required=True, help="Distinct one-letter residues, e.g. AF or EK.")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--reversal-symmetry", choices=("yes", "no"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-only", choices=("yes", "no"), default="no")
    parser.add_argument("--overwrite", choices=("yes", "no"), default="no")
    args = parser.parse_args()
    alphabet = args.amino_acids
    if not alphabet or len(set(alphabet)) != len(alphabet) or set(alphabet) - AMINO_ACIDS:
        parser.error("Specify distinct, uppercase standard amino-acid letters.")
    if args.length <= 0:
        parser.error("--length must be positive.")
    expected = sequence_count(len(alphabet), args.length, args.reversal_symmetry == "yes")
    print(f"Sequences: {expected}", flush=True)
    if args.count_only == "yes":
        return
    output = args.output.expanduser().resolve()
    metadata = output.with_name(output.name + ".metadata.json")
    if args.overwrite != "yes" and (output.exists() or metadata.exists()):
        parser.error("Output or metadata exists; choose a new filename or --overwrite yes.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    digest, count = hashlib.sha256(), 0
    with temporary.open("wb") as handle:
        for sequence in enumerate_sequences(alphabet, args.length, args.reversal_symmetry == "yes"):
            line = (sequence + "\n").encode("ascii")
            handle.write(line)
            digest.update(line)
            count += 1
    if count != expected:
        raise RuntimeError("Enumerated count disagrees with the reversal-orbit counting formula.")
    temporary.replace(output)
    metadata.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "alphabet": "".join(sorted(alphabet)), "length": args.length, "num_sequences": count,
        "reversal_symmetry": args.reversal_symmetry, "ordering": "lexicographic",
        "representative": "lexicographic minimum of sequence and reverse" if args.reversal_symmetry == "yes" else "all oriented sequences",
        "sha256": digest.hexdigest(), "format": "ASCII text; one sequence per line; no header"}, indent=2) + "\n")
    print(f"Wrote {count} sequences: {output}")


if __name__ == "__main__":
    main()
