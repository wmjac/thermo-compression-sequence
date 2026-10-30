"""Feature provenance, human-readable logs, and masked 2D slice plots."""
import csv
import json
import logging
from pathlib import Path
import shlex
import sys
import time
import numpy as np
from _post_training_common_v1 import read_json, sha256


def start_log(directory, command):
    logger = logging.getLogger("pressure_zero_slice")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    file = logging.FileHandler(Path(directory) / "solver.log", mode="w", encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file)
    logger.info("P=0 TWO-DIMENSIONAL SLICE calculation; higher-dimensional coordinates are fixed.")
    logger.info("Exact invocation: %s", shlex.join(command))
    logger.info("Working directory: %s", Path.cwd())
    logger.info("Pressure quantities in the solver are beta P [nm^-3]; concentration is nm^-3.")
    return logger


def dimension_log(parent, directory, d):
    logger = parent.getChild(f"d_{d:03d}")
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    file = logging.FileHandler(Path(directory) / "solver.log", mode="w", encoding="utf-8")
    file.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    file.setFormatter(formatter)
    logger.addHandler(file)
    return logger


def flush_logs(logger):
    for handler in logger.handlers:
        handler.flush()


def load_feature_cloud(features_dir, saved):
    features_dir = Path(features_dir).expanduser().resolve()
    metadata_path = features_dir / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise ValueError(f"Feature export is incomplete: {metadata_path}")
    entries = [row for row in metadata["dimensions"] if row["dimension"] == saved.dimension]
    if len(entries) != 1:
        raise ValueError(f"Feature metadata must contain dimension {saved.dimension} exactly once.")
    entry = entries[0]
    if entry["selection_sha256"] != sha256(saved.best / "selection.json"):
        raise ValueError("Feature TSV belongs to a different selected ensemble.")
    if entry["canonical_config_sha256"] != sha256(saved.canonical / "config.json"):
        raise ValueError("Feature TSV belongs to a different canonicalization.")
    if entry["num_models"] != len(saved.selection["models"]):
        raise ValueError("Feature export model count disagrees with the selected ensemble.")
    path = features_dir / entry["tsv"]
    digest = sha256(path)
    if digest != entry["tsv_sha256"]:
        raise ValueError(f"Feature TSV hash changed: {path}")
    n = metadata["sequence_list"]["num_sequences"]
    d = saved.dimension
    if not isinstance(n, int) or n < 3:
        raise ValueError("At least three exported sequence rows are needed.")
    features = np.empty((n, d), dtype=np.float64)
    rmsd = np.empty(n, dtype=np.float64)
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError("Feature TSV has duplicate column names.")
        wanted = [f"z_{j+1}" for j in range(d)]
        if any(name not in header for name in ["sequence", "feature_rmsd", *wanted]):
            raise ValueError("Feature TSV is missing named z coordinates, sequence, or feature_rmsd.")
        indices = [header.index(name) for name in wanted]
        ri = header.index("feature_rmsd")
        count = 0
        for row in reader:
            if len(row) != len(header) or count >= n:
                raise ValueError(f"Unexpected feature TSV row at line {reader.line_num}.")
            features[count] = [float(row[j]) for j in indices]
            rmsd[count] = float(row[ri])
            count += 1
    if count != n or not np.isfinite(features).all() or not np.isfinite(rmsd).all() or np.any(rmsd < 0):
        raise ValueError("Invalid feature TSV count, values, or RMSD.")
    global_rmsd = float(np.sqrt(np.mean(rmsd**2)))
    return features, global_rmsd, {
        "feature_tsv": str(path), "feature_tsv_sha256": digest,
        "metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path),
        "num_sequences": n, "num_models": entry["num_models"],
        "geometry_mean": "uniform arithmetic mean over the explicitly exported sequence rows",
        "global_rmsd": global_rmsd,
        "global_rmsd_definition": "sqrt(mean_sequence(feature_rmsd**2)), not mean(feature_rmsd)",
        "sequence_list": metadata["sequence_list"],
    }


def render_slice_maps(directory, geometry, axes, retained, root_lists, summary_rows, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    directory = Path(directory)
    shape = geometry["shape"]
    spacing = metadata["voxel_size"]
    points = geometry["projected"]
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    x_edges = np.r_[xs - spacing / 2, xs[-1] + spacing / 2]
    y_edges = np.r_[ys - spacing / 2, ys[-1] + spacing / 2]
    no_root = np.array([retained[i] and not root_lists.get(i) for i in range(len(retained))])
    fixed = ", ".join(f"z_{j}={v:.6g}" for j, v in metadata["fixed_coordinates"].items()) or "none (d=2)"
    # Store exact full-precision coordinates in metadata/TSV; this caption is
    # formatting only and never feeds back into numerical calculations.
    caption = ("Fixed coordinates: " + fixed
               + "\nHull-excluded cells are blank. Black dots: retained points with no positive-slope P=0 root.")
    filenames = []

    def draw(values, filename, title, label, unavailable=None, note=""):
        values = np.asarray(values, dtype=float).copy()
        values[~retained] = np.nan
        image = np.ma.masked_invalid(values.reshape(shape).T)
        fig, ax = plt.subplots(figsize=(8, 6.4))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        finite = values[np.isfinite(values)]
        maximum = float(finite.max()) if len(finite) else 0.
        is_std = filename == "c_tot_model_std.png"
        vmax = (maximum if maximum > 0 else metadata["c_tot_max"] * .01) if is_std else metadata["c_tot_max"]
        mesh = ax.pcolormesh(x_edges, y_edges, image, shading="flat", cmap=cmap,
                             vmin=0., vmax=vmax, rasterized=True)
        if len(finite):
            fig.colorbar(mesh, ax=ax, label=label)
            if is_std and maximum == 0:
                ax.text(.02, .98, "All estimated model SDs are zero", transform=ax.transAxes, va="top", fontsize=8)
        else:
            ax.text(.02, .98, "No values available for this map", transform=ax.transAxes, va="top", fontsize=8)
        if no_root.any():
            ax.scatter(points[no_root, 0], points[no_root, 1], s=9, color="black", zorder=4)
        handles = [Line2D([], [], marker="o", linestyle="", markersize=4, color="black",
                          label="Retained point: no root")]
        if unavailable is not None and np.any(unavailable & retained):
            # Grey cells indicate a root exists but concentration SD cannot be
            # estimated from >=2 unambiguously matched model roots.
            grey = np.ma.masked_where(~(unavailable & retained).reshape(shape).T,
                                     np.ones((shape[1], shape[0])))
            from matplotlib.colors import ListedColormap
            ax.pcolormesh(x_edges, y_edges, grey, cmap=ListedColormap(["#dddddd"]),
                          shading="flat", rasterized=True)
            handles.append(Patch(facecolor="#dddddd", label="Root exists; SD unavailable"))
        if handles:
            # Keep the legend outside the data area so no retained grid
            # centers, particularly no-root dots, are hidden underneath it.
            fig.legend(handles=handles, fontsize=8, loc="lower center",
                       bbox_to_anchor=(.5, .075), ncol=min(2, len(handles)))
        ax.set(xlabel=rf"$\overline{{z}}_{{{axes[0]+1}}}$",
               ylabel=rf"$\overline{{z}}_{{{axes[1]+1}}}$", title=title)
        ax.set_aspect("equal", adjustable="box")
        fig.text(.5, .025, caption + ("\n" + note if note else ""), ha="center", va="bottom", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, .12, 1, 1))
        fig.savefig(directory / filename, dpi=180)
        plt.close(fig)
        filenames.append(filename)

    values = np.full(len(retained), np.nan)
    for i, roots in root_lists.items():
        if roots:
            values[i] = max(root["c_tot"] for root in roots)
    draw(values, "c_tot_largest_root.png",
         f"d={metadata['dimension']}: 2D slice, largest positive-slope ensemble root",
         r"$c_{\mathrm{tot}}$ at $P=0$ (nm$^{-3}$)")
    # All branches remain in roots.tsv. Additional maps use descending local
    # root order; they do not claim global branch continuation.
    max_roots = max((len(rows) for rows in root_lists.values()), default=0)
    for rank in range(2, max_roots + 1):
        branch = np.full(len(retained), np.nan)
        missing = np.zeros(len(retained), dtype=bool)
        for i, roots in root_lists.items():
            ordered = sorted(roots, key=lambda row: row["c_tot"], reverse=True)
            if len(ordered) >= rank:
                branch[i] = ordered[rank - 1]["c_tot"]
            elif ordered:
                missing[i] = True
        # No-root dots retain their precise global meaning. Cells with fewer
        # than rank roots are simply blank in these explicitly ranked maps.
        draw(branch, f"c_tot_root_rank_{rank:02d}.png",
             f"d={metadata['dimension']}: 2D slice, local root rank {rank} (largest first)",
             r"$c_{\mathrm{tot}}$ (nm$^{-3}$)",
             note="Retained cells with fewer roots than this local rank are blank in this ranked map.")
    spread = np.full(len(retained), np.nan)
    for row in summary_rows:
        roots = root_lists[row["point_id"]]
        if row["ensemble_c_tot"] == max(root["c_tot"] for root in roots):
            spread[row["point_id"]] = (np.nan if row["member_root_std"] is None else row["member_root_std"])
    unavailable = retained & ~no_root & ~np.isfinite(spread)
    draw(spread, "c_tot_model_std.png",
         f"d={metadata['dimension']}: 2D slice, model SD near largest ensemble root",
         r"Population SD of matched $c_{\mathrm{tot}}$ roots (nm$^{-3}$)", unavailable)
    return filenames
