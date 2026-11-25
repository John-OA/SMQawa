"""
Photon object building for the photon (gamma+jets) control region.

Development home of build_photons: kept standalone while it is being built so it
can be tested without loading a full processor. When finished it moves into
zz2l2nu_inclusive_test.py (next to build_leptons / build_htaus / build_jets) and
this file is deleted.
"""
import re
import warnings

import awkward as ak
import numpy as np

# ---------------------------------------------------------------------------
# Configuration constants (named, so they can be changed in one place)
# ---------------------------------------------------------------------------
# Photon-lepton cleaning cone, same value build_htaus uses for taus.
# TODO(SMP-23-001): confirm the photon-lepton cleaning cone (0.3 or 0.4).
PHOTON_LEPTON_DR = 0.4

# Kinematics (task brief). Supercluster-eta acceptance and EB/EE gap: same numbers as build_leptons electrons.
PHOTON_PT_MIN = 60.0          # GeV, above the single-photon trigger threshold
PHOTON_SCETA_MAX = 2.5        # tracker/ECAL acceptance
PHOTON_SCETA_EB_MAX = 1.4442  # end of ECAL barrel
PHOTON_SCETA_EE_MIN = 1.5660  # start of ECAL endcap

# POG working points only (EGM provides scale factors for these; no hand-made ID cuts).
# NanoAOD v15 Photon_cutBased: 0 = fail, 1 = loose, 2 = medium, 3 = tight.
PHOTON_ID_WP_CHOICES = ["cutBasedLoose", "cutBasedMedium", "cutBasedTight", "mvaWP90", "mvaWP80"]
PHOTON_ELECTRON_VETO_CHOICES = ["electronVeto", "pixelSeedVeto"]

# NOT DECIDED YET. None = requirement not applied.
# TODO(colleague / SMP-23-001 author): which photon ID working point, and which electron veto (if any)?
PHOTON_ID_WP = None
PHOTON_ELECTRON_VETO = None

# Single-photon HLT paths for the photon control region (OR of all listed paths).
# Only Photon200 is listed: it is the one single-photon path already in 2024-trigger-rules.yaml.
# Consequence: in DATA the region is ~empty below photon pT ~200-220 GeV until lower paths are added.
# TODO(SMP-23-001 author / TSG): Run 3 (2024) single-photon path names below 200 GeV
#   (SMP-23-001 lowest threshold ~50 GeV) and how their PRESCALES were handled (prescale weights?).
PHOTON_TRIGGERS = ["Photon200"]
PHOTON_TRIGGER_DATA_STREAM = "EGamma"  # single-photon paths are recorded in the EGamma primary dataset


def photon_trigger_mask(event, is_mc, triggers=PHOTON_TRIGGERS):
    """
    True for events that fired any single-photon trigger in `triggers`.
    In data, only events from the EGamma stream are accepted, so the same event
    cannot also enter the photon region from another primary dataset (e.g. Muon).
    """
    passed = np.zeros(len(event), dtype=bool)
    for path in triggers:
        if path in event.HLT.fields:
            passed = passed | ak.to_numpy(event.HLT[path])
        else:
            warnings.warn(f"photon_trigger_mask: HLT path {path} not in this file")
    if not is_mc:
        # same stream parsing as common.trigger_rules
        stream = re.split('[_/]', event.metadata['dataset'])[0].rstrip('0123456789')
        if stream != PHOTON_TRIGGER_DATA_STREAM:
            passed = np.zeros(len(event), dtype=bool)
    return passed


def photon_id_mask(photons, wp):
    """True for photons passing the POG ID working point `wp` (None = no ID requirement yet)."""
    if wp is None:
        warnings.warn("build_photons: PHOTON_ID_WP not chosen yet, NO photon ID applied")
        return ak.ones_like(photons.pt, dtype=bool)
    if wp == "cutBasedLoose":
        return photons.cutBased >= 1
    if wp == "cutBasedMedium":
        return photons.cutBased >= 2
    if wp == "cutBasedTight":
        return photons.cutBased >= 3
    if wp == "mvaWP90":
        return photons.mvaID_WP90
    if wp == "mvaWP80":
        return photons.mvaID_WP80
    raise ValueError(f"build_photons: unknown photon ID working point {wp!r}; choices: {PHOTON_ID_WP_CHOICES}")


def photon_electron_veto_mask(photons, veto):
    """True for photons passing the electron veto `veto` (None = no veto yet)."""
    if veto is None:
        warnings.warn("build_photons: PHOTON_ELECTRON_VETO not chosen yet, NO electron veto applied")
        return ak.ones_like(photons.pt, dtype=bool)
    if veto == "electronVeto":   # conversion-safe electron veto
        return photons.electronVeto
    if veto == "pixelSeedVeto":  # stricter: no pixel hits matching the photon
        return ~photons.pixelSeed
    raise ValueError(f"build_photons: unknown electron veto {veto!r}; choices: {PHOTON_ELECTRON_VETO_CHOICES}")


def build_photons(photons, tight_leptons, nanoAODversion="v15", id_wp=PHOTON_ID_WP, electron_veto=PHOTON_ELECTRON_VETO):
    """
    Build the analysis photon collection.

    photons        : event.Photon
    tight_leptons  : tight leptons from build_leptons (muons + electrons)
    nanoAODversion : only "v15" (Run 3) is implemented
    id_wp          : POG photon ID working point (one of PHOTON_ID_WP_CHOICES, or None = not applied yet)
    electron_veto  : one of PHOTON_ELECTRON_VETO_CHOICES, or None = not applied yet

    Returns photons passing the baseline selection that do not overlap a tight lepton,
    sorted by pT (highest first).
    """
    if nanoAODversion not in ["v15"]:
        raise NotImplementedError(f"build_photons: nanoAOD {nanoAODversion} not implemented (only v15)")

    # Baseline selection: each line is one yes/no question per photon; a photon is kept only if all are yes.
    sc_eta = np.abs(photons.superclusterEta)  # where the shower sits in the ECAL (not the vertex-pointed eta)
    baseline_mask = (
        (photons.pt > PHOTON_PT_MIN) &
        ((sc_eta < PHOTON_SCETA_EB_MAX) | ((sc_eta > PHOTON_SCETA_EE_MIN) & (sc_eta < PHOTON_SCETA_MAX))) &  # in barrel or endcap, not in the gap
        photon_id_mask(photons, id_wp) &
        photon_electron_veto_mask(photons, electron_veto)
    )
    # TODO (later step, not now): photon energy scale/smearing (egammaSS.EGM_scale_and_smear, is_electron=False)
    #   and ID/veto scale factors (photon.json.gz) are NOT applied yet.

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
