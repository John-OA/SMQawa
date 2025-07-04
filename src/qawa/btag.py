import correctionlib
import pickle
import hist
import gzip
import os
from hist.intervals import clopper_pearson_interval

from coffea.lookup_tools import dense_lookup
from coffea.lookup_tools import dense_lookup
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.analysis_tools import Weights

import awkward as ak
import numpy as np
import uproot


def btag_id(wp:str='L', era:str='2018'):
    # using deepjet
    # ref : https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation
    dict_wp = {
            "2016"   : {"L": 0.0508, "M": 0.2489, "T": 0.6502},
            "2016APV": {"L": 0.0480, "M": 0.2598, "T": 0.6377},
            "2017"   : {"L": 0.0532, "M": 0.3033, "T": 0.7476},
            "2018"   : {"L": 0.0490, "M": 0.2783, "T": 0.7100}
    }
    return dict_wp[era][wp]

class BTVCorrector:
    def __init__(self, era:str='2018', wp:str='L', tagger:str='deepJet', isAPV=False):
        self._era = era
        self._wp  = wp
        self.isAPV = isAPV
        self._tagger = tagger
        self.eff_hist = None
        _data_path = os.path.join(os.path.dirname(__file__), 'data')
        with gzip.open(
                f"{_data_path}/btv/{era+'_APV' if isAPV else era}_UL"
                f"/eff-btag-{era+'_APV' if isAPV else era}.pkl.gz", 
                'rb') as ifile:
            self.eff_hist = pickle.loads(ifile.read())

        b_tag = self.eff_hist[{'tagger':self._tagger, 'WP': wp, 'passWP': 1  }].values()
        b_all = self.eff_hist[{'tagger':self._tagger, 'WP': wp, 'passWP': sum}].values()

        nom = b_tag / np.maximum(b_all, 1.)
        dw, up = clopper_pearson_interval(b_tag, b_all)

        self.eff        = dense_lookup.dense_lookup(nom,[ax.edges for ax in self.eff_hist.axes[3:]])
        self.eff_statUp = dense_lookup.dense_lookup(up ,[ax.edges for ax in self.eff_hist.axes[3:]])
        self.eff_statDw = dense_lookup.dense_lookup(dw ,[ax.edges for ax in self.eff_hist.axes[3:]])

        self.clib = correctionlib.CorrectionSet.from_file(
                f"{_data_path}/btv/{era+'_APV' if isAPV else era}_UL/"
                f"btagging.json.gz"
        )

        
    def lightSF(self, jet, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        njets = ak.num(jet)
        jet  = ak.flatten(jet)
        sf = self.clib[f"{self._tagger}_incl"].evaluate(
            syst, self._wp,
            np.array(jet.hadronFlavour), 
            np.array(abs(jet.eta)), 
            np.array(jet.pt)
        )
        return ak.unflatten(sf, njets)
    
    def btagSF(self, jet, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        # the jets have to be split between light and c/b quarks
        njets = ak.num(jet)
        jet  = ak.flatten(jet)
        sf = self.clib[f"{self._tagger}_comb"].evaluate(
            syst, self._wp,
            np.array(jet.hadronFlavour),
            np.array(abs(jet.eta)),
            np.array(jet.pt)
        )
        return ak.unflatten(sf, njets)

    def combine(self, eff, sf, b_tagged):
        # tagged SF = SF*eff / eff = SF
        tagged_sf = ak.prod(sf[b_tagged], axis=-1)
        # untagged SF = (1 - SF*eff) / (1 - eff)
        untagged_sf = ak.prod(((1 - sf*eff) / (1 - eff))[~b_tagged], axis=-1)
        return ak.fill_none(tagged_sf * untagged_sf, 1.)

    def append_btag_sf(self, jets: ak.Array, weights: Weights):
        li_jets = jets[(jets.hadronFlavour==0) & (np.abs(jets.eta) <= 2.4) & (jets.pt > 20)]
        bc_jets = jets[(jets.hadronFlavour >0) & (np.abs(jets.eta) <= 2.4) & (jets.pt > 20)]

        wp_era_str = self._era + 'APV' if self.isAPV else self._era
        b_tagged_li = (li_jets.btagDeepFlavB  > btag_id(wp = self._wp, era=wp_era_str))
        b_tagged_bc = (bc_jets.btagDeepFlavB  > btag_id(wp = self._wp, era=wp_era_str))

        eff_li_nom = self.eff(li_jets.hadronFlavour, li_jets.pt, np.abs(li_jets.eta))
        eff_bc_nom = self.eff(bc_jets.hadronFlavour, bc_jets.pt, np.abs(bc_jets.eta))
        eff_li_up  = self.eff_statUp(li_jets.hadronFlavour, li_jets.pt, np.abs(li_jets.eta))
        eff_bc_up  = self.eff_statUp(bc_jets.hadronFlavour, bc_jets.pt, np.abs(bc_jets.eta))
        eff_li_dw  = self.eff_statDw(li_jets.hadronFlavour, li_jets.pt, np.abs(li_jets.eta))
        eff_bc_dw  = self.eff_statDw(bc_jets.hadronFlavour, bc_jets.pt, np.abs(bc_jets.eta))

        sf_nom_li = self.combine(eff_li_nom, self.lightSF(li_jets, 'central'), b_tagged_li)
        sf_nom_bc = self.combine(eff_bc_nom, self.btagSF(bc_jets, 'central'), b_tagged_bc)
        sf_nom = sf_nom_li * sf_nom_bc

        # nominal scale factors
        weights.add('btag_sf_light', sf_nom_li)
        weights.add('btag_sf_bc'   , sf_nom_bc)
        weights.add('btag_sf_nom'  , sf_nom)

        # statistical uncertainties
        weights.add(
            f'btag_sf_stat',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(
                eff_li_up, self.lightSF(li_jets, 'central'), b_tagged_li
            ) * self.combine(
                eff_bc_up, self.btagSF(bc_jets, 'central'), b_tagged_bc
            ) / (sf_nom_li*sf_nom_bc),
            weightDown= self.combine(
                eff_li_dw, self.lightSF(li_jets, 'central'), b_tagged_li
            ) * self.combine(
                eff_bc_dw, self.btagSF(bc_jets, 'central'), b_tagged_bc
            ) / (sf_nom_li*sf_nom_bc)
        )

        # uncorrelated unceertainties
        weights.add(
            f'btag_sf_light_{self._era + "APV" if self.isAPV else self._era}',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(eff_li_nom, self.lightSF(li_jets, 'up'), b_tagged_li),
            weightDown= self.combine(eff_li_nom, self.lightSF(li_jets, 'down'), b_tagged_li)
        )
        weights.add(
            f'btag_sf_bc_{self._era + "APV" if self.isAPV else self._era}',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(eff_bc_nom, self.btagSF(bc_jets, 'up'), b_tagged_bc),
            weightDown= self.combine(eff_bc_nom, self.btagSF(bc_jets, 'down'), b_tagged_bc)
        )

        # correlated uncertainties
        weights.add(
            'btag_sf_light_correlated',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(eff_li_nom, self.lightSF(li_jets, 'up_correlated'), b_tagged_li),
            weightDown= self.combine(eff_li_nom, self.lightSF(li_jets, 'down_correlated'), b_tagged_li)
        )
        weights.add(
            'btag_sf_bc_correlated',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(eff_bc_nom, self.btagSF(bc_jets, 'up_correlated'), b_tagged_bc),
            weightDown= self.combine(eff_bc_nom, self.btagSF(bc_jets, 'down_correlated'), b_tagged_bc)
        )

        return sf_nom
