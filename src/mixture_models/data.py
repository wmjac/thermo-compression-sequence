"""Physical data records; sequence length and tokenization are model-owned."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .units import UnitSystem


@dataclass(frozen=True)
class MixtureRecord:
    uid: str
    source: str
    source_index: int
    system_id: str
    sequences: tuple[str, ...]
    counts: tuple[int, ...]
    concentrations: tuple[float, ...]
    ctot: float
    pressure: float
    nominal_temperature_k: float
    observables: dict


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_files(paths):
    """Pool systems in CLI file order, then file order; no sequence-length assumptions."""
    records, manifests, seen_hashes = [], [], set()
    seen_source_ids = set()
    for filename in paths:
        path = Path(filename).expanduser().resolve()
        digest = sha256_file(path)
        if digest in seen_hashes:
            raise ValueError(f"Duplicate input JSON content: {path}")
        seen_hashes.add(digest)
        with path.open() as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: dataset JSON must be an object with units and systems.")
        try:
            units = UnitSystem.from_block(raw.get("units"))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        systems = raw.get("systems")
        if not isinstance(systems, list) or not systems:
            raise ValueError(f"{path}: systems must be a nonempty list.")
        manifests.append({"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size,
                          "num_systems": len(systems), "units": units.declarations,
                          "nominal_temperature_k": units.temperature_k,
                          "conversion_factors": {"concentration": units.concentration,
                                                 "pressure": units.pressure,
                                                 "potential_energy": units.potential_energy}})
        for index, system in enumerate(systems):
            system_id = str(system.get("directory_name", index))
            context = f"{path}: system {system_id} (index {index})"
            try:
                props = system["input_data_header"]["mixture_properties"]
                chains = props["chains"]
                if not isinstance(chains, list) or not chains:
                    raise ValueError("chains must be a nonempty list of [sequence, count].")
                sequences, counts = [], []
                for sequence, count in chains:
                    if not isinstance(sequence, str) or not sequence:
                        raise ValueError("Every sequence must be a nonempty string.")
                    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                        raise ValueError("Chain counts must be positive integers.")
                    sequences.append(sequence)
                    counts.append(count)
                if len(set(sequences)) != len(sequences):
                    raise ValueError("Repeated sequence entries must be combined in the source JSON.")
                if "total chains" in props and props["total chains"] != sum(counts):
                    raise ValueError("total chains disagrees with the chain counts.")
                ctot = float(props["chain concentration"]) * units.concentration
                if not math.isfinite(ctot) or ctot <= 0:
                    raise ValueError("Total chain concentration must be finite and positive.")
                if "monomer concentration" in props:
                    monomer = float(props["monomer concentration"]) * units.concentration
                    expected = ctot * sum(len(s) * n for s, n in zip(sequences, counts)) / sum(counts)
                    if not math.isclose(monomer, expected, rel_tol=1e-6, abs_tol=1e-14):
                        raise ValueError("Monomer and chain concentrations disagree with sequence lengths/counts.")
                obs = {name: units.observable(name, value) for name, value in system["observables"].items()}
                pressure = float(obs["pressure"]["mean"])
                if not math.isfinite(pressure):
                    raise ValueError("Pressure mean must be finite.")
                for name in ("pressure", "pe", "temperature"):
                    for field in ("mean", "stddev", "sem"):
                        if name in obs and field in obs[name] and not math.isfinite(float(obs[name][field])):
                            raise ValueError(f"Nonfinite {name}.{field}.")
                        if name in obs and field in ("stddev", "sem") and float(obs[name].get(field, 0.0)) < 0:
                            raise ValueError(f"Negative {name}.{field}.")
                identity = (digest, system_id)
                if identity in seen_source_ids:
                    raise ValueError("Repeated system identifier within the source file.")
                seen_source_ids.add(identity)
                concentrations = tuple(ctot * count / sum(counts) for count in counts)
                records.append(MixtureRecord(f"{digest}:{index}", str(path), index, system_id,
                                             tuple(sequences), tuple(counts), concentrations,
                                             ctot, pressure, units.temperature_k, obs))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{context}: {exc}") from exc
    return records, manifests


def validate_with_model(records, module):
    for record in records:
        try:
            module.validate_sequences(record.sequences)
        except ValueError as exc:
            raise ValueError(f"{record.source}: system {record.system_id}: {exc}") from exc


def load_run_records(run):
    """Verify archived inputs before reusing training records in other tasks."""
    from .provenance import read_json
    run = Path(run)
    manifests = read_json(run / "inputs.json")
    records, actual = load_json_files([run / item["snapshot"] for item in manifests])
    for expected, current in zip(manifests, actual):
        if expected["sha256"] != current["sha256"]:
            raise ValueError(f"Archived training JSON has changed: {current['path']}")
    return records, manifests


def save_physical_dataset(path, records):
    """Archive converted physical values in float64 independently of training dtype."""
    offsets = np.concatenate(([0], np.cumsum([len(r.sequences) for r in records])))
    observables = {}
    for name in ("pressure", "pe", "temperature"):
        for field in ("mean", "stddev", "sem"):
            observables[f"{name}_{field}"] = np.array([r.observables.get(name, {}).get(field, np.nan)
                                                       for r in records], dtype=np.float64)
    np.savez_compressed(path, uid=np.array([r.uid for r in records]),
                        source=np.array([r.source for r in records]),
                        system_id=np.array([r.system_id for r in records]),
                        offsets=offsets, sequences=np.array([s for r in records for s in r.sequences]),
                        counts=np.array([n for r in records for n in r.counts], dtype=np.int64),
                        concentrations=np.array([c for r in records for c in r.concentrations], dtype=np.float64),
                        ctot=np.array([r.ctot for r in records], dtype=np.float64),
                        pressure=np.array([r.pressure for r in records], dtype=np.float64),
                        nominal_temperature_k=np.array([r.nominal_temperature_k for r in records]),
                        **observables)
