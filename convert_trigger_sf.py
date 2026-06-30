#!/usr/bin/env python
"""Convert the legacy ROOT trigger-efficiency scale factors used by the
``inc-WZ`` processor into correctionlib (schema v2) JSON files.

Background
----------
For Run 2 (2016/2017/2018) the WZ inclusive processor
(:mod:`qawa.process.wztau2lnu_inclusive`) reads its di-lepton trigger scale
factors from ``data/trigger_sf/histo_triggerEff_sel0_<era>.root``. Each file
holds **16 TH2D** histograms named ``trgSF{FLAVOR}{ETAREGION}`` with

* ``FLAVOR``    in {EE, EM, ME, MM} -- the (leading, subleading) lepton pair,
* ``ETAREGION`` in {BB, BE, EB, EE} -- the barrel/endcap region of each lepton
  (first letter = leading lepton, second = subleading),

each a 7x7 grid over (leading pt, subleading pt) with bin edges
``[20, 25, 30, 35, 40, 50, 60, 70]``. The grids store **scale factors**
(eff_data / eff_mc; values span ~0.43-1.12) plus a per-bin Gaussian error, and
are lower triangular: only bins with ``lead_pt >= subl_pt`` are filled (the rest
are zero), matching the physical ordering enforced by the analysis.

For 2024 the processor instead consumes a correctionlib file
``data/trigger_sf/triggerSF_2024.json`` exposing **4 flavor-only** corrections
``SF_mm``/``SF_me``/``SF_em``/``SF_ee``, each a
``category(systematic) -> multibinning(lead_pt, trail_pt)`` with the same pt
edges, evaluated with **no eta** information:
``SF_xx.evaluate(lead_pt, subl_pt, systematic)``.

This script reproduces that flavor-only schema for the Run 2 ROOT files. Since
the source files store only scale factors (not the numerator/denominator counts
of the underlying efficiencies), the four eta regions cannot be re-summed from
raw counts. They are instead combined per (flavor, lead_pt, subl_pt) bin with an
**inverse-variance weighting** -- the best available proxy for count-weighting
given only SF +/- error:

    w_r   = 1 / sigma_r^2                       (over filled regions r)
    SF    = sum_r (w_r * SF_r) / sum_r w_r
    sigma = sqrt(1 / sum_r w_r)

Only regions whose bin is filled (sigma_r > 0) contribute; bins unfilled in all
regions (the unphysical ``lead_pt < subl_pt`` corner) stay zero, mirroring the
legacy ROOT padding.

The resulting grid is laid out exactly as the legacy lookup reads it, so that
``SF_xx.evaluate(lead_pt, subl_pt, syst)`` reproduces the combined value
bin-for-bin:

* ``content[i_lead * n_trail + i_trail] = grid[i_lead, i_trail]``
* ``up   = SF + sigma`` / ``down = SF - sigma`` (symmetric, as in the legacy
  weight which uses ``center +/- error``)
* ``flow = "clamp"`` so pt above the last edge clamps to the last bin, matching
  the legacy ``digitize`` whose open last bin extends to 1e5.

The eta dependence of the source histograms is **not** preserved; see
``test_trigger_sf.py`` which checks that the combined correctionlib output
matches the inverse-variance combination of the legacy per-region lookups.

Usage
-----
    python convert_trigger_sf.py                       # all eras
    python convert_trigger_sf.py --era 2018
    python convert_trigger_sf.py --eras 2016 2017 2018 --outdir src/qawa/data/trigger_sf
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import uproot
from correctionlib import schemav2 as cs

# (leading, subleading) flavor pairs and the correctionlib correction names the
# processor evaluates. The ROOT histograms are named trgSF<FLAVOR><ETAREGION>.
FLAVORS = ("EE", "EM", "ME", "MM")
CORRECTION_NAME = {"EE": "SF_ee", "EM": "SF_em", "ME": "SF_me", "MM": "SF_mm"}
ETA_REGIONS = ("BB", "BE", "EB", "EE")

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "src", "qawa", "data", "trigger_sf"
)


def root_path(era: str, data_dir: str = _DEFAULT_DATA_DIR) -> str:
    return os.path.join(data_dir, f"histo_triggerEff_sel0_{era}.root")


def combine_regions(values, errors):
    """Inverse-variance combination of the four eta regions, per pt-pt bin.

    Parameters
    ----------
    values, errors : sequence of four (7, 7) arrays
        Scale factor and Gaussian error for regions BB, BE, EB, EE.

    Returns
    -------
    (sf, sigma) : two (7, 7) arrays
        Combined scale factor and its error. Bins unfilled in every region
        (error == 0 everywhere) are set to zero, matching the legacy padding.
    """
    V = np.stack([np.asarray(v, dtype=float) for v in values])   # (4, 7, 7)
    E = np.stack([np.asarray(e, dtype=float) for e in errors])   # (4, 7, 7)
    filled = E > 0
    w = np.where(filled, 1.0 / np.where(filled, E, 1.0) ** 2, 0.0)
    wsum = w.sum(axis=0)
    have = wsum > 0
    sf = np.where(have, (w * V).sum(axis=0) / np.where(have, wsum, 1.0), 0.0)
    sigma = np.where(have, np.sqrt(1.0 / np.where(have, wsum, 1.0)), 0.0)
    return sf, sigma


def _multibinning(edges_x, edges_y, content: np.ndarray) -> cs.MultiBinning:
    """Build a MultiBinning over (lead_pt, trail_pt) from a 7x7 grid."""
    return cs.MultiBinning(
        nodetype="multibinning",
        inputs=["lead_pt", "trail_pt"],
        edges=[list(map(float, edges_x)), list(map(float, edges_y))],
        # C-order flatten: index = i_lead * n_trail + i_trail
        content=[float(v) for v in np.asarray(content).ravel(order="C")],
        flow="clamp",
    )


def build_correction(flavor: str, root_file) -> cs.Correction:
    """Build one flavor-only correction by combining the four eta regions."""
    hists = {r: root_file[f"trgSF{flavor}{r}"] for r in ETA_REGIONS}
    ref = hists["BB"]
    edges_x = ref.axes[0].edges()
    edges_y = ref.axes[1].edges()

    values = [hists[r].values() for r in ETA_REGIONS]
    errors = [np.sqrt(hists[r].variances()) for r in ETA_REGIONS]
    sf, sigma = combine_regions(values, errors)

    return cs.Correction(
        name=CORRECTION_NAME[flavor],
        description=(
            f"Di-lepton trigger scale factor {CORRECTION_NAME[flavor]} "
            f"(leading={flavor[0]}, subleading={flavor[1]}), converted from the "
            f"four trgSF{flavor}<region> histograms via per-bin inverse-variance "
            f"combination over eta regions BB/BE/EB/EE (eta dependence collapsed)."
        ),
        version=1,
        inputs=[
            cs.Variable(name="lead_pt", type="real", description="Leading lepton pT [GeV]"),
            cs.Variable(name="trail_pt", type="real", description="Trailing lepton pT [GeV]"),
            cs.Variable(name="systematic", type="string", description="nominal / up / down"),
        ],
        output=cs.Variable(name="weight", type="real", description="Scale factor weight"),
        data=cs.Category(
            nodetype="category",
            input="systematic",
            content=[
                cs.CategoryItem(key="nominal", value=_multibinning(edges_x, edges_y, sf)),
                cs.CategoryItem(key="up", value=_multibinning(edges_x, edges_y, sf + sigma)),
                cs.CategoryItem(key="down", value=_multibinning(edges_x, edges_y, sf - sigma)),
            ],
        ),
    )


def build_correction_set(era: str, data_dir: str = _DEFAULT_DATA_DIR) -> cs.CorrectionSet:
    """Build the 4-correction CorrectionSet (SF_mm/me/em/ee) for one era."""
    with uproot.open(root_path(era, data_dir)) as root_file:
        corrections = [build_correction(fl, root_file) for fl in FLAVORS]
    return cs.CorrectionSet(
        schema_version=2,
        description=(
            f"WZ inclusive di-lepton trigger scale factors for {era}, converted "
            f"from histo_triggerEff_sel0_{era}.root. Flavor-only schema matching "
            f"triggerSF_2024.json; the four eta regions are combined per pt-pt bin "
            f"by inverse-variance weighting (eta dependence is NOT preserved)."
        ),
        corrections=corrections,
    )


def convert(era: str, outdir: str = _DEFAULT_DATA_DIR,
            data_dir: str = _DEFAULT_DATA_DIR) -> str:
    cset = build_correction_set(era, data_dir=data_dir)
    out_path = os.path.join(outdir, f"triggerSF_{era}.json")
    with open(out_path, "w") as fout:
        fout.write(cset.model_dump_json(exclude_unset=True, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eras", nargs="+", default=["2016", "2017", "2018"],
                        help="Eras to convert (default: 2016 2017 2018)")
    parser.add_argument("--era", default=None, help="Single era (overrides --eras)")
    parser.add_argument("--outdir", default=_DEFAULT_DATA_DIR,
                        help="Output directory for the JSON files")
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                        help="Directory holding the ROOT input files")
    args = parser.parse_args()

    eras = [args.era] if args.era else args.eras
    for era in eras:
        out_path = convert(era, outdir=args.outdir, data_dir=args.data_dir)
        print(f"[{era}] wrote {out_path}")


if __name__ == "__main__":
    main()
