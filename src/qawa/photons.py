"""
Photon object building for the photon (gamma+jets) control region.

Development home of build_photons: kept standalone while it is being built so it
can be tested without loading a full processor. When finished it moves into
zz2l2nu_inclusive_test.py (next to build_leptons / build_htaus / build_jets) and
this file is deleted.
"""
import awkward as ak
import numpy as np  # noqa: F401  (used by the Step 2 selection)

# ---------------------------------------------------------------------------
# Configuration constants (named, so they can be changed in one place)
# ---------------------------------------------------------------------------
# Photon-lepton cleaning cone, same value build_htaus uses for taus.
# TODO(SMP-23-001): confirm the photon-lepton cleaning cone (0.3 or 0.4).
PHOTON_LEPTON_DR = 0.4


def build_photons(photons, tight_leptons, nanoAODversion="v15"):
    """
    Build the analysis photon collection.

    photons        : event.Photon
    tight_leptons  : tight leptons from build_leptons (muons + electrons)
    nanoAODversion : only "v15" (Run 3) is implemented

    Returns photons that do not overlap a tight lepton, sorted by pT (highest first).
    """
    if nanoAODversion not in ["v15"]:
        raise NotImplementedError(f"build_photons: nanoAOD {nanoAODversion} not implemented (only v15)")

    # Step 2: baseline selection (POG working point, pT, eta) replaces this "keep everything" mask
    baseline_mask = ak.ones_like(photons.pt, dtype=bool)

    # Cleaning: drop a photon if any tight lepton is within DeltaR < PHOTON_LEPTON_DR,
    # because that photon is most likely the same ECAL deposit as the lepton (electron)
    # or radiation from it (muon FSR). The lepton is kept, the photon is removed.
    overlap_leptons = ak.any(
        photons.metric_table(tight_leptons) <= PHOTON_LEPTON_DR,
        axis=2
    )

    good_photons = photons[baseline_mask & ~overlap_leptons]

    # Sort by pT so ak.firsts(good_photons) is really the leading photon,
    # even after energy corrections (TODO, later step) change the order.
    good_photons = good_photons[ak.argsort(good_photons.pt, axis=1, ascending=False)]

    return good_photons
