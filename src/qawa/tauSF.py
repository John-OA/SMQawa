import correctionlib
import pickle
import hist
import gzip
import os

from coffea.analysis_tools import Weights

import awkward as ak
import numpy as np
import uproot

# from correctionlib import _core

class tauIDScaleFactors:
    def __init__(self, era:str='2018', vsjet_wp:str='VTight', vse_wp:str='VVLoose', vsmu_wp:str='VLoose',
                 isAPV=False, isEE=False, isBPix=False):

        config = {"2016": {"tagger": "DeepTau2017v2p1"},
                  "2017": {"tagger": "DeepTau2017v2p1"},
                  "2018": {"tagger": "DeepTau2017v2p1"},
                  "2022": {"tagger": "DeepTau2018v2p5"},
                  # "2023": {"tagger": "DeepTau2018v2p5"}, # Maybe we want UParT tagger or Run 3, however!
                  # "2024": {"tagger": "DeepTau2018v2p5"},
                  # "2025": {"tagger": "DeepTau2018v2p5"},
                  }
        self._era = era
        self.vsjet_wp = vsjet_wp
        self.vse_wp = vse_wp
        self.vsmu_wp = vsmu_wp
        self.isAPV = isAPV
        self.isEE = isEE
        self.isBPix = isBPix
        assert sum([self.isAPV, self.isEE, self.isBPix]) <= 1, "Multiple incompatible suberas selected as active"
        if isAPV:
            self._subera = "_APV"
        elif isEE:
            self._subera = "_EE"
        elif isBPix:
            self._subera = "_BPix"
        else:
            self._subera = ""
        self.tagger = config[self._era]["tagger"]
        _data_path = os.path.join(os.path.dirname(__file__), 'data')
        # Load CorrectionSet
        fname = f"{_data_path}/tau_ID_SF/{self._era+self._subera}/tau.json.gz"
        with gzip.open(fname,'rt') as file:
            data = file.read().strip()
            cset = correctionlib.CorrectionSet.from_string(data)
            # cset = correctionlib.CorrectionSet.from_file(file)

        #Load Correction Objects :
        self.corr_vsjet = cset[f"{self.tagger}VSjet"]
        self.corr_vse = cset[f"{self.tagger}VSe"]
        self.corr_vsmu = cset[f"{self.tagger}VSmu"]
        self.corr_enscale = cset["tau_energy_scale"]


    def getSF(self, tau, syst="nom"):

        ntaus = ak.num(tau)
        taus  = ak.flatten(tau)
        tau_eta = np.array(taus.eta)
        tau_pt  = np.array(taus.pt)
        tau_dm  = np.array(taus.decayMode)
        tau_genmatch = np.array(taus.genPartFlav)

        # {self.tagger}VSjet
        # sf_vsjet = self.corr_vsjet.evaluate(tau_pt,tau_dm,tau_genmatch, self.vsjet_wp, self.vse_wp, syst,"pt")
        sf_vsjet = self.corr_vsjet.evaluate(tau_pt,tau_dm,tau_genmatch, self.vsjet_wp, "VVLoose", syst,"pt") 
        sf_vsjet = ak.fill_none(sf_vsjet, 1.0)
        sf_vsjet = ak.unflatten(sf_vsjet, ntaus)
        sf_vsjet = ak.prod(sf_vsjet, axis=-1)

        # {self.tagger}VSe
        sf_vse = self.corr_vse.evaluate(tau_eta,tau_genmatch,self.vse_wp,syst)
        sf_vse = ak.fill_none(sf_vse, 1.)
        sf_vse = ak.unflatten(sf_vse, ntaus)
        sf_vse = ak.prod(sf_vse, axis=-1)

        # {self.tagger}VSmu
        sf_vsmu = self.corr_vsmu.evaluate(tau_eta,tau_genmatch,self.vsmu_wp,syst)
        sf_vsmu = ak.fill_none(sf_vsmu, 1.)
        sf_vsmu = ak.unflatten(sf_vsmu, ntaus)
        sf_vsmu = ak.prod(sf_vsmu, axis=-1)

        # tau energy scale
        sf_enscale = self.corr_enscale.evaluate(tau_pt,tau_eta,tau_dm,tau_genmatch,self.tagger,syst)
        sf_enscale = ak.fill_none(sf_enscale, 1.)
        sf_enscale = ak.unflatten(sf_enscale, ntaus)
        sf_enscale = ak.prod(sf_enscale, axis=-1)

        return  sf_vsjet, sf_vse, sf_vsmu


    def apply_function_flattened_masked(self,func, *args, flatten_axis=1, valid_where=None):
        maybe_flat_args = []
        num = None
        for arg in args:
            if isinstance(arg, ak.Array):
                if valid_where is None:
                    valid_where = ak.ones_like(arg, dtype=bool)
                if num is None:
                    num = ak.num(arg)
        maybe_flat_args = [ak.flatten(ak.mask(arg, valid_where), axis=flatten_axis) if isinstance(arg, ak.Array) else arg for arg in args]
        return ak.unflatten(
            func(*maybe_flat_args),
            num
        )

    def tau_energy_scale_correction(self, tau):

        # Usage, lets pretend only the dm 5 and 6 had to be avoided but all other inputs were valid
        valid_tau_enscale = (tau.decayMode != 5) & (tau.decayMode != 6)

        # ntaus = ak.num(tau)
        # taus  = ak.flatten(tau)
        tau_eta = tau.eta
        tau_pt  = tau.pt
        tau_mass = tau.mass
        tau_dm  = tau.decayMode
        tau_genmatch = tau.genPartFlav

        enscale_nom_with_none = self.apply_function_flattened_masked(
            self.corr_enscale.evaluate,
            tau_pt,
            tau_eta,
            tau_dm,
            tau_genmatch,
            self.tagger,
            "nom",
            valid_where = valid_tau_enscale
        )
        enscale_up_with_none = self.apply_function_flattened_masked(
            self.corr_enscale.evaluate,
            tau_pt,
            tau_eta,
            tau_dm,
            tau_genmatch,
            self.tagger,
            "up",
            valid_where = valid_tau_enscale
        )
        enscale_down_with_none = self.apply_function_flattened_masked(
            self.corr_enscale.evaluate,
            tau_pt,
            tau_eta,
            tau_dm,
            tau_genmatch,
            self.tagger,
            "down",
            valid_where = valid_tau_enscale
        )
        #now enscale_nom should have a SF where the valid_tau_enscale is true, and "None" elsewhere, so fill_none it with 1.0 to not alter the scale
        enscale_nom = ak.fill_none(enscale_nom_with_none, 1.0)
        enscale_up = ak.fill_none(enscale_up_with_none, 1.0)
        enscale_down = ak.fill_none(enscale_down_with_none, 1.0)

        # ak.num(enscale_nom, axis=1) == ak.num(tau.pt, axis=1)
        # ak.num(enscale_up, axis=1) == ak.num(tau.pt, axis=1)
        # ak.num(enscale_down, axis=1) == ak.num(tau.pt, axis=1)

        tau_pt = enscale_nom*tau_pt
        tau_pt_EnUp = enscale_up*tau_pt
        tau_pt_EnDown = enscale_down*tau_pt

        tau_mass = enscale_nom*tau_mass
        tau_mass_EnUp = enscale_up*tau_mass
        tau_mass_EnDown = enscale_down*tau_mass

        return tau_pt, tau_pt_EnUp, tau_pt_EnDown, tau_mass, tau_mass_EnUp, tau_mass_EnDown

    def append_tauID_sf(self, taus: ak.Array, weights: Weights):

        # taus = taus[(taus.pt >= 20) & (np.abs(taus.eta) <= 2.3) & (np.abs(taus.dz) < 0.2)]

        sf_vsjet_nom, sf_vse_nom, sf_vsmu_nom = self.getSF(taus, syst="nom")
        sf_vsjet_up, sf_vse_up, sf_vsmu_up = self.getSF(taus, syst="up")
        sf_vsjet_down, sf_vse_down, sf_vsmu_down = self.getSF(taus, syst="down")


        weights.add('tauIDvsjet_sf', sf_vsjet_nom, sf_vsjet_up, sf_vsjet_down)
        weights.add('tauIDvse_sf'  , sf_vse_nom, sf_vse_up, sf_vse_down)
        weights.add('tauIDvsmu_sf' , sf_vsmu_nom, sf_vsmu_up, sf_vsmu_down)

        return sf_vsjet_nom, sf_vsjet_up, sf_vsjet_down, sf_vse_nom, sf_vse_up, sf_vse_down, sf_vsmu_nom, sf_vsmu_up, sf_vsmu_down

