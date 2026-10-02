#!/usr/bin/env python3
"""
NeurIPS figure 5 — Spectral-feedback sequences (max edit positions / maxspecorder 10) folded with
ESMFold vs native PDBs.

Loads three spectral-feedback eval CSVs under ``fmif/eval_results/test/target/group1/``, matching
these **underlying models**:

  - **pretrained** — pretrained checkpoint, **BoN=1** (``bon_N=1`` in the CSV filename)
  - **bon10** — pretrained checkpoint, **BoN=10**
  - **drakes** — **DRAKES** model, **BoN=1**

By default only **2KRU** and **r6** (CSV ``r6_560_TrROS_Hall.pdb``, outputs labeled ``r6_650_TrROS_Hall``)
are folded — six combinations (three models × two proteins). Use ``--all-spectral-proteins`` for
all targets in the CSVs.

PyMOL: black background, green native / red prediction, oblique camera (no labels or margins).

Dependencies: PyMOL on PATH for PNGs (``--no-pymol`` to skip). CUDA GPU strongly recommended.

Examples:
  python neurips_figures5.py
  python neurips_figures5.py --all-spectral-proteins
  python neurips_figures5.py --spectral-runs pretrained,drakes --no-pymol
  python neurips_figures5.py --eval-csv path/to/run.csv --single-run-label ablation1
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import biotite.structure as struc
import numpy as np
import pandas as pd
import torch
import esm
from biotite.structure.io.pdb import PDBFile

try:
    from biotite.structure import superimpose_apply as _superimpose_apply
except ImportError:
    _superimpose_apply = None

_PYRO_INIT = False

# Default CSV ``protein_name`` values (native PDBs under AlphaFold_model_PDBs/)
DEFAULT_FIG5_PROTEINS = "2KRU.pdb,r6_560_TrROS_Hall.pdb"


def protein_display_stem(protein_name: str) -> str:
    """Stable filename stem; maps eval PDB name to paper-style label where needed."""
    stem = Path(protein_name).stem
    if stem == "r6_560_TrROS_Hall":
        return "r6_650_TrROS_Hall"
    return stem.replace(".", "_")


def expand_run_specs(args: argparse.Namespace, fmif: Path) -> list[tuple[str, Path]]:
    """Map run id -> CSV: pretrained BoN=1, pretrained BoN=10, drakes BoN=1 (spectral, maxspecorder=10)."""
    subdir = fmif / "eval_results/test/target/group1"
    defaults: dict[str, Path] = {
        "pretrained": subdir
        / "pretrained_test_ddg_bon_N=1_feedbacksteps=5_feedbackmethod=spectral_maxspecorder=10_masks=8192.csv",
        "bon10": subdir
        / "pretrained_test_ddg_bon_N=10_feedbacksteps=5_feedbackmethod=spectral_maxspecorder=10_masks=8192_rmax=False.csv",
        "drakes": subdir / "drakes_test_ddg_bon_N=1_feedbacksteps=5_feedbackmethod=spectral_maxspecorder=10_masks=8192.csv",
    }
    overrides: dict[str, Path | None] = {
        "pretrained": args.pretrained_csv,
        "bon10": args.bon_csv,
        "drakes": args.drakes_csv,
    }
    if args.eval_csv is not None:
        label = (args.single_run_label or "custom").strip() or "custom"
        return [(label, args.eval_csv.resolve())]

    parts = [s.strip() for s in args.spectral_runs.split(",") if s.strip()]
    unknown = [s for s in parts if s not in defaults]
    if unknown:
        raise ValueError(f"Unknown --spectral-runs {unknown}; allowed: {sorted(defaults)}")

    out: list[tuple[str, Path]] = []
    for key in parts:
        p = overrides[key]
        path = Path(p).resolve() if p is not None else defaults[key]
        out.append((key, path.resolve()))
    return out


def iter_selected_rows(
    sub: pd.DataFrame, *, row_mode: str, all_rows: bool
) -> list[tuple[str, pd.Series]]:
    """If all_rows, yield every table row for this protein; else one row via pick_row."""
    if sub.empty:
        return []
    if all_rows:
        return [(f"rep{i}", row) for i, (_, row) in enumerate(sub.iterrows())]
    return [("", pick_row(sub, row_mode))]


def ensure_native_relaxed(
    protein_name: str,
    pdb_dir: Path,
    workdir: str,
    no_relax: bool,
    cache: dict[str, str],
) -> str:
    """Copy native PDB into workdir, optionally relax once per protein."""
    if protein_name in cache:
        return cache[protein_name]
    stem = Path(protein_name).stem
    native_src = pdb_dir / protein_name
    if not native_src.is_file():
        raise FileNotFoundError(f"Native PDB not found: {native_src}")
    out = os.path.join(workdir, f"native_relaxed_{stem}.pdb")
    shutil.copy(native_src, out)
    if not no_relax:
        if not relax_pdb_like_scrmsd(out):
            print("  (PyRosetta relax skipped for native — not installed)")
    else:
        print("  (--no-relax: skipping native relax)")
    cache[protein_name] = out
    return out


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--repo",
        type=Path,
        default=here,
        help="DRAKES repo root (default: directory containing this script)",
    )
    p.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="Single eval CSV instead of the default triple (pretrained / bon10 / drakes).",
    )
    p.add_argument(
        "--spectral-runs",
        type=str,
        default="pretrained,bon10,drakes",
        help="Comma-separated if not using --eval-csv: pretrained (pretrained BoN=1), bon10 "
        "(pretrained BoN=10), drakes (drakes BoN=1).",
    )
    p.add_argument("--pretrained-csv", type=Path, default=None, metavar="PATH")
    p.add_argument("--bon-csv", type=Path, default=None, metavar="PATH")
    p.add_argument("--drakes-csv", type=Path, default=None, metavar="PATH")
    p.add_argument(
        "--single-run-label",
        type=str,
        default="custom",
        help="Filename prefix when using --eval-csv (default: custom).",
    )
    p.add_argument(
        "--only-proteins",
        type=str,
        default=DEFAULT_FIG5_PROTEINS,
        metavar="NAMES",
        help=f"Comma-separated CSV protein_name values (default: {DEFAULT_FIG5_PROTEINS}).",
    )
    p.add_argument(
        "--all-spectral-proteins",
        action="store_true",
        help="Fold every protein in each CSV instead of the default 2KRU + r6 pair.",
    )
    p.add_argument(
        "--all-rows",
        action="store_true",
        help="Fold every CSV row per protein (e.g. replicates), not just one (--row).",
    )
    p.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory of native PDBs (default: .../AlphaFold_model_PDBs)",
    )
    p.add_argument(
        "--pt-hub",
        type=Path,
        default=None,
        help="torch.hub cache dir (default: data_and_model/.cache/torch/)",
    )
    p.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Output directory for PDF overlays (default: <repo>/neurips_fig5_pdfs)",
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Temp working directory for intermediate PDBs (default: system temp prefix)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device, e.g. cuda:0 or cpu (default: cuda:0 if available else cpu)",
    )
    p.add_argument(
        "--no-relax",
        action="store_true",
        help="Skip PyRosetta relax (only ESMFold + alignment + PDFs)",
    )
    p.add_argument(
        "--try-3d",
        action="store_true",
        help="Try py3Dmol interactive view (usually only useful in Jupyter; ignored in most CLIs)",
    )
    p.add_argument(
        "--row",
        choices=("first", "min_scrmsd"),
        default="first",
        help="Which row to take when multiple rows exist per protein (default: first)",
    )
    p.add_argument(
        "--no-pymol",
        action="store_true",
        help="Skip PyMOL ray-traced PNG export (default: run PyMOL if executable is found)",
    )
    p.add_argument("--pymol-width", type=int, default=3200, help="PyMOL ray width in px")
    p.add_argument("--pymol-height", type=int, default=3200, help="PyMOL ray height in px")
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    fmif = repo / "DRAKES" / "drakes_protein" / "fmif"
    pdb_dir = args.pdb_dir
    if pdb_dir is None:
        pdb_dir = repo / "DRAKES/data/data_and_model/proteindpo_data/AlphaFold_model_PDBs"
    pt_hub = args.pt_hub
    if pt_hub is None:
        pt_hub = repo / "DRAKES/data/data_and_model/.cache/torch/"
    pdf_dir = args.pdf_dir
    if pdf_dir is None:
        pdf_dir = repo / "neurips_fig5_pdfs"
    return {
        "repo": repo,
        "fmif": fmif,
        "pdb_dir": pdb_dir.resolve(),
        "pt_hub": pt_hub.resolve(),
        "pdf_dir": pdf_dir.resolve(),
    }


def pick_row(sub: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "first":
        return sub.iloc[0]
    if "scrmsd" not in sub.columns:
        return sub.iloc[0]
    return sub.loc[sub["scrmsd"].idxmin()]


def fold_sequence_esmfold(
    seq: str,
    out_pdb: str,
    device: str,
    model: Any,
    pt_hub: Path,
) -> tuple[Any, float]:
    seq = seq.replace("X", "A")
    if model is None:
        pt_hub.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(pt_hub))
        torch_dev = torch.device(device)
        model = esm.pretrained.esmfold_v1().eval().to(torch_dev)
        # FP16 LayerNorm is not implemented on CPU in PyTorch; esmfold may load FP16 weights.
        if torch_dev.type == "cpu":
            model = model.float()
    with torch.no_grad():
        out = model.infer(seq)
        pdb_str = model.output_to_pdb(out)[0]
    with open(out_pdb, "w") as f:
        f.write(pdb_str)
    plddt = out["mean_plddt"][0].item()
    return model, plddt


def relax_pdb_like_scrmsd(pdb_path: str) -> bool:
    global _PYRO_INIT
    try:
        import pyrosetta
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
    except ImportError:
        return False
    if not _PYRO_INIT:
        pyrosetta.init(extra_options="-out:level 100")
        _PYRO_INIT = True
    scorefxn = pyrosetta.create_score_function("ref2015_cart")
    pose = pyrosetta.pose_from_file(pdb_path)
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(pose))
    packer.apply(pose)
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.apply(pose)
    pose.dump_pdb(pdb_path)
    return True


def align_prediction_to_native(native_pdb: str, pred_pdb: str, out_pdb: str) -> None:
    nat = PDBFile.read(native_pdb).get_structure(model=1)
    pred = PDBFile.read(pred_pdb).get_structure(model=1)
    nat_ca = nat[nat.atom_name == "CA"]
    pred_ca = pred[pred.atom_name == "CA"]
    L = min(len(nat_ca), len(pred_ca))
    if len(nat_ca) != len(pred_ca):
        print(f"  CA count mismatch native={len(nat_ca)} pred={len(pred_ca)} — using first {L} residues")
    nat_ca = nat_ca[:L]
    pred_ca = pred_ca[:L]
    _, trafo = struc.superimpose(nat_ca, pred_ca)
    pred_copy = pred.copy()
    # Older biotite (e.g. multiflow env): trafo is tuple (t1, rot, t2) → use superimpose_apply.
    # Newer biotite: trafo is AffineTransformation → .apply().
    if isinstance(trafo, tuple):
        if _superimpose_apply is None:
            raise RuntimeError(
                "biotite superimpose returned a tuple transform but superimpose_apply is missing"
            )
        pred_aligned = _superimpose_apply(pred_copy, trafo)
    else:
        pred_aligned = trafo.apply(pred_copy)
    file = PDBFile()
    file.set_structure(pred_aligned)
    file.write(out_pdb)


def show_native_vs_pred(native_pdb: str, pred_aligned_pdb: str, title: str) -> None:
    try:
        import py3Dmol
    except ImportError:
        print("py3Dmol not installed; skipping 3D view.")
        print("  native:", native_pdb, "\n  pred: ", pred_aligned_pdb)
        return
    with open(native_pdb) as f:
        nat = f.read()
    with open(pred_aligned_pdb) as f:
        pred = f.read()
    view = py3Dmol.view(width=700, height=480)
    view.addModel(nat, "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "#bbbbbb"}})
    view.addModel(pred, "pdb")
    view.setStyle({"model": 1}, {"cartoon": {"color": "#2ecc71"}})
    view.zoomTo()
    print(title)
    view.show()


def _ca_coords(pdb_path: str) -> np.ndarray:
    atoms = PDBFile.read(pdb_path).get_structure(model=1)
    ca = atoms[atoms.atom_name == "CA"]
    return ca.coord.copy()


def save_ca_overlay_pdf(
    native_pdb: str,
    pred_aligned_pdb: str,
    pdf_path: str,
    title: str,
    dpi: int = 150,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nat = _ca_coords(native_pdb)
    pred = _ca_coords(pred_aligned_pdb)
    L = min(len(nat), len(pred))
    nat = nat[:L]
    pred = pred[:L]
    pts = np.vstack([nat, pred])

    fig = plt.figure(figsize=(5.5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(nat[:, 0], nat[:, 1], nat[:, 2], color="#888888", linewidth=1.4, label="Native")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="#27ae60", linewidth=1.4, label="ESMFold (spectral seq.)")

    cr = pts.min(axis=0)
    cl = pts.max(axis=0)
    ctr = (cr + cl) / 2
    span = (cl - cr).max() / 2 + 1e-6
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    pdf_path = os.path.abspath(pdf_path)
    d = os.path.dirname(pdf_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved PDF: {pdf_path}")


def _find_pymol_executable() -> str | None:
    import shutil

    for name in ("pymol", "pymol2"):
        path = shutil.which(name)
        if path:
            return path
    return None


def render_pymol_overlay_png(
    native_pdb: str,
    pred_pdb: str,
    out_png: str,
    *,
    ray_w: int = 3200,
    ray_h: int = 3200,
) -> bool:
    """
    Ray-traced PyMOL PNG: both structures on a black background only (no labels or margins).

    Uses an oblique camera (yaw/pitch/roll) after ``orient`` so the fold reads in depth rather than head-on.

    Requires the `pymol` or `pymol2` CLI (e.g. conda-forge `pymol-open-source`).
    """
    pymol_bin = _find_pymol_executable()
    if not pymol_bin:
        print("PyMOL executable not found in PATH; skipping ray-traced PNG.", file=sys.stderr)
        print("  Install e.g.: conda install -c conda-forge pymol-open-source", file=sys.stderr)
        return False

    native_pdb = os.path.abspath(native_pdb)
    pred_pdb = os.path.abspath(pred_pdb)
    out_png = os.path.abspath(out_png)
    out_dir = os.path.dirname(out_png)
    png_basename = os.path.basename(out_png)
    # PyMOL's `png` always appends `.png`; passing `foo.png` yields `foo.png.png` (see ScenePNG line).
    png_stem = Path(out_png).stem
    os.makedirs(out_dir or ".", exist_ok=True)

    def esc(p: str) -> str:
        return p.replace("\\", "/").replace("'", "\\'")

    # Batch PyMOL often drops absolute paths for `png` and writes to cwd instead.
    # cd into the target directory and save with a basename only (most reliable).
    # Black viewport only; oblique POV (not straight-on) after orient; PyMOL appends .png to filename stem.
    pml = f"""
# NeurIPS fig5 — native vs ESMFold (spectral sequence)
reinitialize
load '{esc(native_pdb)}', nat
load '{esc(pred_pdb)}', pred
hide everything
show cartoon
color tv_green, nat
color tv_red, pred
set cartoon_smooth_loops, 1
set cartoon_fancy_helices, 1
bg_color black
set ray_opaque_background, 1
set orthoscopic, 1
set depth_cue, 1
set ray_trace_mode, 1
set ray_shadows, 1
set ambient, 0.42
set direct, 0.58
set reflect, 1.0
set antialias, 2
orient
zoom complete=1
turn y, 62
turn x, -36
turn z, 22
move z, -14
zoom complete=0.91
ray {ray_w}, {ray_h}
cd '{esc(out_dir)}'
png {png_stem}
quit
"""
    fd, pml_path = tempfile.mkstemp(suffix=".pml", prefix="neurips_fig5_pymol_")
    proc = None
    try:
        with os.fdopen(fd, "w") as f:
            f.write(pml)
        proc = subprocess.run(
            [pymol_bin, "-cq", pml_path],
            cwd=out_dir,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            print("PyMOL exited with code", proc.returncode, file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            if proc.stdout:
                print(proc.stdout, file=sys.stderr)
            return False
    finally:
        try:
            os.unlink(pml_path)
        except OSError:
            pass

    launch_cwd = os.getcwd()
    # Expected path after fix: <stem>.png. Older buggy runs wrote <basename>.png i.e. *.png.png.
    candidates = [
        out_png,
        os.path.join(out_dir, png_stem + ".png"),
        os.path.join(out_dir, png_basename + ".png"),
        os.path.join(launch_cwd, png_stem + ".png"),
        os.path.join(launch_cwd, png_basename),
        os.path.join(launch_cwd, png_basename + ".png"),
    ]
    found = next((p for p in candidates if os.path.isfile(p)), None)
    if found and os.path.abspath(found) != os.path.abspath(out_png):
        shutil.move(found, out_png)
        print(f"Saved PyMOL ray trace (relocated): {out_png}")
        return True
    if os.path.isfile(out_png):
        print(f"Saved PyMOL ray trace: {out_png}")
        return True

    print(f"PyMOL finished but PNG missing: {out_png}", file=sys.stderr)
    if proc is not None and proc.stdout:
        print("--- PyMOL stdout ---", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
    if proc is not None and proc.stderr:
        print("--- PyMOL stderr ---", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
    print(
        "Tip: batch PyMOL often needs a display or OSMesa; try: xvfb-run -a python ... "
        "or conda install -c conda-forge pymol-open-source (with libegl/OSMesa).",
        file=sys.stderr,
    )
    return False


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    pdb_dir = paths["pdb_dir"]
    pt_hub = paths["pt_hub"]
    pdf_dir = paths["pdf_dir"]
    fmif = paths["fmif"]

    if not pdb_dir.is_dir():
        print(f"ERROR: PDB directory not found: {pdb_dir}", file=sys.stderr)
        return 1

    try:
        run_specs = expand_run_specs(args, fmif)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    for run_id, csv_path in run_specs:
        if not csv_path.is_file():
            print(f"ERROR: eval CSV not found for run {run_id!r}: {csv_path}", file=sys.stderr)
            return 1

    if args.all_spectral_proteins:
        only = None
    else:
        only = {x.strip() for x in args.only_proteins.split(",") if x.strip()}

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        workdir = str(workdir)
    else:
        workdir = tempfile.mkdtemp(prefix="neurips_fig5_")
    print("working dir:", workdir)

    native_cache: dict[str, str] = {}
    esm_model = None
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for run_id, csv_path in run_specs:
        df = pd.read_csv(csv_path)
        if "protein_name" not in df.columns or "seq" not in df.columns:
            print(f"ERROR: CSV missing protein_name/seq: {csv_path}", file=sys.stderr)
            return 1
        proteins = sorted(df["protein_name"].unique())
        if only is not None:
            proteins = [p for p in proteins if p in only]
            missing = only - set(proteins)
            if missing:
                print(f"WARNING: --only-proteins not found in {run_id} CSV: {missing}", file=sys.stderr)

        print(f"--- run {run_id} ({csv_path.name}) — {len(proteins)} proteins ---")

        for protein_name in proteins:
            sub = df[df["protein_name"] == protein_name]
            rows_pack = iter_selected_rows(sub, row_mode=args.row, all_rows=args.all_rows)
            stem_safe = protein_display_stem(protein_name)

            try:
                native_for_align = ensure_native_relaxed(
                    protein_name, pdb_dir, workdir, args.no_relax, native_cache
                )
            except FileNotFoundError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1

            for rep_suffix, row in rows_pack:
                seq = row["seq"]
                scr = row["scrmsd"] if "scrmsd" in row.index else None
                tag = f"{stem_safe}_{rep_suffix}" if rep_suffix else stem_safe
                print(f"  {run_id}/{tag}: n={len(seq)} scrmsd={scr}")

                raw_pred = os.path.join(workdir, f"{run_id}_{tag}_pred_raw.pdb")
                aln_pred = os.path.join(workdir, f"{run_id}_{tag}_pred_aligned.pdb")

                esm_model, plddt = fold_sequence_esmfold(seq, raw_pred, device, esm_model, pt_hub)
                if not args.no_relax:
                    ok = relax_pdb_like_scrmsd(raw_pred)
                    if not ok:
                        print("    (PyRosetta relax skipped for prediction — not installed)")
                else:
                    print("    (--no-relax: skipping prediction relax)")

                align_prediction_to_native(native_for_align, raw_pred, aln_pred)
                print(f"    mean pLDDT ≈ {plddt:.2f}")

                base = f"{run_id}_{tag}".replace(" ", "_").replace("/", "-")
                caption = (
                    f"{stem_safe} [{run_id}] — native vs ESMFold (spectral seq., maxspecorder=10), "
                    f"mean pLDDT={plddt:.2f}"
                )
                pdf_path = pdf_dir / f"{base}_esmfold_overlay.pdf"
                save_ca_overlay_pdf(
                    native_for_align,
                    aln_pred,
                    str(pdf_path),
                    title=caption,
                )
                if not args.no_pymol:
                    pymol_png = pdf_dir / f"{base}_esmfold_pymol.png"
                    render_pymol_overlay_png(
                        native_for_align,
                        aln_pred,
                        str(pymol_png),
                        ray_w=args.pymol_width,
                        ray_h=args.pymol_height,
                    )
                if args.try_3d:
                    show_native_vs_pred(
                        native_for_align,
                        aln_pred,
                        title=f"{base} pLDDT={plddt:.2f}",
                    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
