import awkward as ak
import numpy as np
import scipy.interpolate as interp
from scipy import stats as st
from scipy.special import expit
import uproot
import pickle
import hist
import yaml
import os
import re
import onnxruntime as rt
import numpy.lib.recfunctions as rf
import uproot
import pickle
import numpy as np
import pandas as pd
import math

from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from qawa.roccor import rochester_correction
from qawa.leptonsSF import LeptonScaleFactors
from qawa.jetPU import jetPUScaleFactors
from qawa.tauSF import tauIDScaleFactors
from qawa.btag import BTVCorrector, btag_id
from qawa.jme import JMEUncertainty, update_collection
from qawa.gen_match import find_best_match
from qawa.datadriven_variation import DataDrivenEventReweight
from qawa.common import pileup_weights, ewk_corrector, met_phi_xy_correction, theory_ps_weight, theory_pdf_weight, trigger_rules, transverse_energy




def build_leptons(muons, electrons):
    # select tight/loose muons
    tight_muons_mask = (
        (muons.pt             >  20. ) &
        (np.abs(muons.eta)    <  2.4 ) &
        (np.abs(muons.dxy)    <  0.045) &
        (np.abs(muons.dz )    <  0.2 ) &
        (muons.pfRelIso04_all <= 0.15) & 
        muons.tightId
    )
    tight_muons = muons[tight_muons_mask]
    loose_muons = muons[
        ~tight_muons_mask &
        (muons.pt            >  10. ) &
        (np.abs(muons.eta)   <  2.4 ) &
        (muons.pfRelIso04_all<= 0.25) &
        muons.looseId   
    ]
    non_iso_muons = muons[
        ~tight_muons_mask &
        (muons.pt            >  10. ) &
        (np.abs(muons.eta)   <  2.4 ) &
        (muons.pfRelIso04_all > 0.25) &
        muons.looseId   
    )
    non_iso_muons = muons[non_iso_muons_mask]
    # select tight/loose electron
    # https://cms-talk.web.cern.ch/t/clarification-on-ee-eb-gap-veto/133256
    electron_superclusterEta = electrons.eta + electrons.deltaEtaSC
    tight_electrons_mask = (
        (electrons.pt           > 20.) &
        ((np.abs(electron_superclusterEta) < 1.4442) | ((np.abs(electron_superclusterEta) > 1.5660) & (np.abs(electron_superclusterEta)  < 2.5)))  &
        electrons.mvaFall17V2Iso_WP90
    )
    tight_electrons = electrons[tight_electrons_mask]

    loose_electrons_mask = (
        ~tight_electrons_mask & 
        (electrons.pt           > 10.) &
        (np.abs(electrons.eta)  < 2.5)  &
        electrons.mvaFall17V2Iso_WPL
    )
    loose_electrons = electrons[loose_electrons_mask]

    non_iso_electrons_mask = (
        ~tight_electrons_mask & 
        ~loose_electrons_mask &
        (electrons.pt           > 10.) &
        (np.abs(electrons.eta)  < 2.5)  &
        electrons.mvaFall17V2noIso_WPL
    )

    non_iso_electrons = electrons[non_iso_electrons_mask]

    # contruct a lepton object
    tight_leptons = ak.with_name(ak.concatenate([tight_muons, tight_electrons], axis=1), 'PtEtaPhiMCandidate')
    loose_leptons = ak.with_name(ak.concatenate([loose_muons, loose_electrons], axis=1), 'PtEtaPhiMCandidate')
    non_iso_leptons = ak.with_name(ak.concatenate([non_iso_muons, non_iso_electrons], axis=1), 'PtEtaPhiMCandidate')

    return tight_leptons, loose_leptons, non_iso_leptons

def build_htaus(tau, lepton):
    #print(dir(tau))
    #print(tau.__dict__)
    
    base_selection = (
        (tau.pt         > 20 ) & 
        (np.abs(tau.eta)< 2.3 ) &
        (np.abs(tau.dz)< 0.2 ) &
        (tau.decayMode != 5   ) & 
        (tau.decayMode != 6   ) &
        (tau.idDeepTau2017v2p1VSe >= 64) &
        (tau.idDeepTau2017v2p1VSmu >= 8) &
        (tau.idDeepTau2017v2p1VSjet >= 64)
    )

    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )
   
    return tau[base_selection & ~overlap_leptons]

def build_htaus_tight(tau, lepton):
    #print(dir(tau))
    #print(tau.__dict__)
    
    base_selection = (
        (tau.pt         > 20 ) & 
        (np.abs(tau.eta)< 2.3 ) & 
        (np.abs(tau.dz)< 0.2 ) &
        (tau.decayMode != 5   ) & 
        (tau.decayMode != 6   ) &
        (tau.idDeepTau2017v2p1VSe >= 32) &
        (tau.idDeepTau2017v2p1VSmu >= 4) &
        (tau.idDeepTau2017v2p1VSjet >= 32)
    )

    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )
    
    return tau[base_selection & ~overlap_leptons]

def build_htaus_loose(tau, lepton):
    #print(dir(tau))
    #print(tau.__dict__)
    
    base_selection = (
        (tau.pt         > 20 ) & 
        (np.abs(tau.eta)< 2.3 ) & 
        (np.abs(tau.dz)< 0.2 ) &
        (tau.decayMode != 5   ) & 
        (tau.decayMode != 6   ) &
        (tau.idDeepTau2017v2p1VSe >= 1) &
        (tau.idDeepTau2017v2p1VSmu >= 1) &
        (tau.idDeepTau2017v2p1VSjet >= 1)
    )

    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )
    
    return tau[base_selection & ~overlap_leptons]


def build_jets(jets, tight_leptons, taus_loose, btag_wp, era, isAPV):

    overlap_leptons = ak.any(jets.metric_table(tight_leptons) <= 0.4, axis=2)
    overlap_taus = ak.any(jets.metric_table(taus_loose) <= 0.4, axis=2)


    jet_mask = (
            ~overlap_leptons & 
            ~overlap_taus &
            (jets.pt>30.0) & 
            (np.abs(jets.eta) < 4.7) & 
            (jets.jetId >= 6) & # tight JetID 7(2016) and 6(2017/8)
            ((jets.puId >= 6) | (jets.puId == 3) | (jets.pt >= 50)) # medium puID https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL 3,7 for 16and 16APV; 6,7 for 17,18
        )

    jet_btag = (
                jets.btagDeepFlavB > btag_id(
                    btag_wp, 
                    era + 'APV' if isAPV else era
                ) 
        ) & (np.abs(jets.eta)<2.4)

    return jets[jet_mask], jets[jet_mask & jet_btag] 

def apply_hem_uncertainty(jets, met):
    
    phi_mask = (
        (jets.phi > -1.57) &
        (jets.phi < -0.87)
    )
    eta_mask_20 = (jets.eta > -2.5) & (jets.eta < -1.3)
    eta_mask_35 = (jets.eta > -3.0) & (jets.eta < -2.5)
    
    mask_20 = phi_mask & eta_mask_20
    mask_35 = phi_mask & eta_mask_35

    scale = ak.ones_like(jets.pt)
    scale = ak.where(mask_20, 0.80, scale)
    scale = ak.where(mask_35, 0.65, scale)

    scaled_jets = ak.with_field(jets, jets.pt * scale, 'pt')
    scaled_jets = ak.with_field(scaled_jets, jets.mass * scale, 'mass')

    delta_px = ak.sum((jets.pt - scaled_jets.pt) * np.cos(jets.phi), axis=1, mask_identity=False) #delta is negative vector sum of corrected jet_pt minus old jet_pt
    delta_py = ak.sum((jets.pt - scaled_jets.pt) * np.sin(jets.phi), axis=1, mask_identity=False)

    met_px = met.pt * np.cos(met.phi)
    met_py = met.pt * np.sin(met.phi)
    shifted_px = met_px + delta_px
    shifted_py = met_py + delta_py
    shifted_pt = np.sqrt(shifted_px**2 + shifted_py**2)
    shifted_phi = np.arctan2(shifted_py, shifted_px)

    scaled_met = ak.with_field(met, shifted_pt, 'pt')
    scaled_met = ak.with_field(scaled_met, shifted_phi, 'phi')

    return scaled_jets, scaled_met


class wzinclusive_processor(processor.ProcessorABC):
    # EWK corrections process has to be define before hand, it has to change when we move to dask
    def __init__(self, era: str ='2018', ewk_process_name=None, run_period: str = ''): 
        self._era = era
        if 'APV' in self._era:
            self._isAPV = True
            self._era = re.findall(r'\d+', self._era)[0] 
            #print(f"[YACINE DEBUG] era={self._era} APV={self._isAPV}")
        else:
            self._isAPV = False

        
        
        
        jec_tag = ''
        jer_tag = ''
        if len(run_period)==0:
            if self._era == '2016':
                if self._isAPV:
                    jec_tag = 'Summer19UL16APV_V7_MC'
                    jer_tag = 'Summer20UL16APV_JRV3_MC'
                else:
                    jec_tag = 'Summer19UL16_V7_MC'
                    jer_tag = 'Summer20UL16_JRV3_MC'
            elif self._era == '2017':
                jec_tag = 'Summer19UL17_V5_MC'
                jer_tag = 'Summer19UL17_JRV2_MC'
            elif self._era == '2018':
                jec_tag = 'Summer19UL18_V5_MC'
                jer_tag = 'Summer19UL18_JRV2_MC'
            elif self._era in ["2022", "2023", "2024", "2025"]:
                raise NotImplementedError(f"{self._era} is not yet implemented")
            else:
                raise ValueError(f"Unexpected era={self._era}")
        else:
            if self._era == '2016':
                if self._isAPV:
                    if run_period in ['B', 'C', 'D']:
                        jec_tag = 'Summer19UL16APV_RunBCD_V7_DATA'
                    else:
                        jec_tag = 'Summer19UL16APV_RunEF_V7_DATA'
                else:
                    jec_tag = 'Summer19UL16_RunFGH_V7_DATA'
            elif self._era == '2017':
                jec_tag = f'Summer19UL17_Run{run_period}_V5_DATA'
            elif self._era == '2018':
                jec_tag = f'Summer19UL18_Run{run_period}_V5_DATA'
            elif self._era in ["2022", "2023", "2024", "2025"]:
                raise NotImplementedError(f"{self._era} is not yet implemented")
            else:
                raise ValueError(f"Unexpected era={self._era}")
        
        self.btag_wp = 'L'
        self.jetPU_wp = 'M'
        self.tauIDvsjet_wp = 'VTight' #Medium is working
        self.tauIDvse_wp = 'VTight' #changing to Vloose from VVLoose
        self.tauIDvsmu_wp = 'Tight'
        self.zmass = 91.1873 # GeV 
        self._btag = BTVCorrector(era=self._era, wp=self.btag_wp, isAPV=self._isAPV)
        self._jmeu = JMEUncertainty(jec_tag, jer_tag, era=self._era, is_mc=(len(run_period)==0))
        self._purw = pileup_weights(era=self._era)
        self._leSF = LeptonScaleFactors(era=self._era, isAPV=self._isAPV)
        self._jpSF = jetPUScaleFactors(era=self._era, wp=self.jetPU_wp, isAPV=self._isAPV)
        self._tauID= tauIDScaleFactors(era=self._era, vsjet_wp=self.tauIDvsjet_wp,vse_wp=self.tauIDvse_wp, vsmu_wp=self.tauIDvsmu_wp, isAPV=self._isAPV)
        self._dd   = DataDrivenEventReweight(era=self._era)

        _data_path = 'qawa/data'
        _data_path = os.path.join(os.path.dirname(__file__), '../data')
        self._json = {
            '2018': LumiMask(f'{_data_path}/json/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt'),
            '2017': LumiMask(f'{_data_path}/json/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
            '2016': LumiMask(f'{_data_path}/json/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'),
        }
        with open(f'{_data_path}/{self._era}-trigger-rules.yaml') as ftrig:
            self._triggers = yaml.load(ftrig, Loader=yaml.FullLoader)
            
        with open(f'{_data_path}/eft-names.dat') as eft_file:
            self._eftnames = [n.strip() for n in eft_file.readlines()]

        with uproot.open(f'{_data_path}/trigger_sf/histo_triggerEff_sel0_{self._era}.root') as _fn:
            _hvalue = np.dstack([_fn[_hn].values() for _hn in _fn.keys()] + [np.ones((7,7))])
            _herror = np.dstack([np.sqrt(_fn[_hn].variances()) for _hn in _fn.keys()] + [np.zeros((7,7))])
            self.trig_sf_map = np.stack([_hvalue, _herror], axis=-1)

        self.ewk_process_name = ewk_process_name
        if self.ewk_process_name is not None:
            self.ewk_corr = ewk_corrector(process=ewk_process_name)

        self.ABCD_tau_bins = [20,25,30,35,40,60,80,100,1000]
        #to change this in tau_pt use this in tau histogram hist.axis.Variable(self.ABCD_tau_bins, name="tau_pt", label=r"$p_{T}^{tau}$ (GeV)")

        self.build_histos = lambda: {
            'dilep_mt_llnunu': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="dilep_mt_llnunu", label=r"$M_{T}^{\ell\ell}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'dilep_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="dilep_pt", label=r"$p_{T}^{\ell\ell}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'dilep_deta': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(20, 0, 8, name="dilep_deta", label=r"$Delta\eta_{\ell\ell}$"),
                hist.storage.Weight()
            ),
            'HTl': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="HTl", label=r"$H_{Tl}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'ST': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="ST", label=r"$S_{T}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'dilep_tau_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="dilep_tau_pt", label=r"$p_{T}^{\tau \ell\ell}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'dilep_loose_tau_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="dilep_loose_tau_pt", label=r"$p_{T}^{\ell\ell, \tau}$ (GeV)"),
                hist.storage.Weight()
            ),
            'dilep_m': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 180, name="dilep_m", label=r"$M_{\ell\ell}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'met_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="met_pt", label=r"$p_{T}^{miss}$ (GeV)"),
                hist.storage.Weight()
            ),
            'mT_W': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="mT_W", label=r"$m_{T}^{W}$ (GeV)"),
                hist.storage.Weight()
            ),
            'dilep_tau_loose_met_hadron_mt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(150, 0, 1500, name="dilep_tau_loose_met_hadron_mt", label=r"$M_{T}^{WZ}$ (GeV)"),
                hist.storage.Weight()
            ),
            'mT_WZ': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(150, 0, 1500, name="mT_WZ", label=r"$m_{T}^{WZ}$ (GeV)"),
                hist.storage.Weight()
            ),
            'inv_m_WZ': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(150, 0, 1500, name="inv_m_WZ", label=r"$m_{inv}^{WZ}$ (GeV)"),
                hist.storage.Weight()
            ),
            'tau_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(60, 0, 600, name="tau_pt", label=r"$p_{T}^{tau}$ (GeV)"), 
                # hist.axis.Variable(self.ABCD_tau_bins, name="tau_pt", label=r"$p_{T}^{tau}$ (GeV)"),
                hist.storage.Weight()
            ),
            'taus_eta': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="taus_eta", label=r"$\eta(\tau)$"),
                hist.storage.Weight()
            ),
            'taus_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="taus_phi", label=r"$\phi(\tau)$"),
                hist.storage.Weight()
            ),
            'tau_pt_loose': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(60, 0, 600, name="tau_pt_loose", label=r"$p_{T}^{tau_loose}$ (GeV)"), 
                # hist.axis.Variable(self.ABCD_tau_bins, name="tau_pt_loose", label=r"$p_{T}^{tau_loose}$ (GeV)"),
                hist.storage.Weight()
            ),
            'taus_eta_loose': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="taus_eta_loose", label=r"$\eta(\tau loose)$"),
                hist.storage.Weight()
            ),
            'taus_phi_loose': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="taus_phi_loose", label=r"$\phi(\tau loose)$"),
                hist.storage.Weight()
            ),
            'met_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="met_phi", label=r"$\phi(p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            'delta_tau_met_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="delta_tau_met_phi", label=r"$\Delta \phi(\tau, p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            'deep_tau_jet': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1, name="deep_tau_jet", label=r"$deep_{\tau}^{jet}$"),
                hist.storage.Weight()
            ),
            'deep_tau_e': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1, name="deep_tau_e", label=r"$deep_{\tau}^{e}$"),
                hist.storage.Weight()
            ),
            'deep_tau_mu': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1, name="deep_tau_mu", label=r"$deep_{\tau}^{mu}$"),
                hist.storage.Weight()
            ),
            'dphi_met_ll': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="dphi_met_ll", label=r"$\Delta \phi(\ell\ell,p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            'dphi_jet_met': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="dphi_jet_met", label=r"$\Delta \phi(j,p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            'dilep_dphi_tau': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_dphi_tau", label=r"$\Delta \phi(\ell\ell,\tau)$"),
                hist.storage.Weight()
            ),
            'dilep_loose_tau_met_dphi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_loose_tau_met_dphi", label=r"$\Delta \phi(\ell\ell\tau, p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            'dilep_loose_tau_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_loose_tau_phi", label=r"$\phi(\ell\ell\tau)$"),
                hist.storage.Weight()
            ),
            'dilep_tau_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_tau_phi", label=r"$\phi(\ell\ell\tau)$"),
                hist.storage.Weight()
            ),
            'dilep_dphi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_dphi", label=r"$\Delta \phi(\ell\ell)$"),
                hist.storage.Weight()
            ),
            'baseweight': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(400, -100, 100, name="baseweight", label=r"baseweight"),
                hist.storage.Weight()
            ),
            'delta_R': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, 5, name="delta_R", label=r"$\Delta R (ll, \tau)$"),
                hist.storage.Weight()
            ),
            'lead_jet_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="lead_jet_pt", label="$p_T^{j_1}$ (GeV)"),
                hist.storage.Weight()
            ),
            'lead_jet_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="lead_jet_phi", label=r"$\phi($p_T^{j_1})$"),
                hist.storage.Weight()
            ),
            'lead_jet_eta': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="lead_jet_eta", label=r"$\eta(j_1)$"),
                hist.storage.Weight()
            ), 
            'delta_R_jet_dilep': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="delta_R_jet_dilep", label=r"$\Delta R(\ell\ell,j)$"),
                hist.storage.Weight()
            ),
            'delta_R_jet_tau': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="delta_R_jet_tau", label=r"$\Delta R(\tau,j)$"),
                hist.storage.Weight()
            ),
            'dilep_dR': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="dilep_dR", label=r"$\Delta R(\ell\ell)$"),
                hist.storage.Weight()
            ),
            'min_dphi_met_j': hist.Hist( 
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="min_dphi_met_j", label=r"$\min\Delta\phi(p_{T}^{miss},j)$"),
                hist.storage.Weight()
            ),
            'leading_lep_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="leading_lep_pt", label="$p_T^{l_1}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'trailing_lep_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="trailing_lep_pt", label=r"$p_T^{l_2}$ (GeV)"),
                hist.storage.Weight()
            ),
            'leading_lep_eta': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="leading_lep_eta", label=r"$\eta(l_1)$"),
                hist.storage.Weight()
            ), 
            'trailing_lep_eta': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="trailing_lep_eta", label=r"$\eta(l_2)$"),
                hist.storage.Weight()
            ),
            'leading_lep_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="leading_lep_phi", label=r"$\phi^(l_1)$"),
                hist.storage.Weight()
            ), 
            'trailing_lep_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="trailing_lep_phi", label=r"$\phi^(l_2)$"),
                hist.storage.Weight()
            ),
            'njets': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="njets", label=r"$N_{jet}$ ($p_{T}>30$ GeV)"),
                hist.storage.Weight()
            ), 
            'nbjets': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="nbjets", label=r"$N_{b-jet}$ ($p_{T}>30$ GeV)"),
                hist.storage.Weight()
            ),
            'ntaus_vtight': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="ntaus_vtight", label=r"$N_{taus}$ (vtight)"),
                hist.storage.Weight()
            ), 
            'ntaus_tight': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="ntaus_tight", label=r"$N_{taus}$ (tight)"),
                hist.storage.Weight()
            ),
            'ntaus_loose': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="ntaus_loose", label=r"$N_{taus}$ (loose)"),
                hist.storage.Weight()
            ),
            'delta_R_non_iso_lep_loose_tau': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="delta_R_non_iso_lep_loose_tau", label=r"$\Delta R(\tau (loose),non_iso_lep)$"),
                hist.storage.Weight()
            ),
            'delta_R_non_iso_lep_vtight_tau': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="delta_R_non_iso_lep_vtight_tau", label=r"$\Delta R(\tau (loose),tightlep)$"),
                hist.storage.Weight()
            ),
            'delta_R_non_iso_lep_tight_tau': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="delta_R_non_iso_lep_tight_tau", label=r"$\Delta R(\tau (loose),looselep)$"),
                hist.storage.Weight()
            ),
        }

    
    def _add_trigger_sf(self, weights, lead_lep, subl_lep):
        mask_BB = ak.fill_none((lead_lep.eta <= 1.5) & (subl_lep.eta <= 1.5), False)
        mask_EB = ak.fill_none((lead_lep.eta >= 1.5) & (subl_lep.eta <= 1.5), False)
        mask_BE = ak.fill_none((lead_lep.eta <= 1.5) & (subl_lep.eta >= 1.5), False)
        mask_EE = ak.fill_none((lead_lep.eta >= 1.5) & (subl_lep.eta >= 1.5), False)

        mask_mm = ak.fill_none((np.abs(lead_lep.pdgId)==13) & (np.abs(subl_lep.pdgId)==13), False)
        mask_ee = ak.fill_none((np.abs(lead_lep.pdgId)==11) & (np.abs(subl_lep.pdgId)==11), False)
       
        mask_me = (~mask_mm & ~mask_ee) & (np.abs(lead_lep.pdgId) == 13)
        mask_em = (~mask_mm & ~mask_ee) & (np.abs(lead_lep.pdgId) == 11)

        lept_pt_bins = [20, 25, 30, 35, 40, 50, 60, 100000]
        lep_1_bin = np.digitize(lead_lep.pt.to_numpy(), lept_pt_bins) - 1
        lep_2_bin = np.digitize(subl_lep.pt.to_numpy(), lept_pt_bins) - 1
        trigg_bin = np.select([
            (mask_ee & mask_BB).to_numpy(),
            (mask_ee & mask_BE).to_numpy(),
            (mask_ee & mask_EB).to_numpy(),
            (mask_ee & mask_EE).to_numpy(),

            (mask_em & mask_BB).to_numpy(),
            (mask_em & mask_BE).to_numpy(),
            (mask_em & mask_EB).to_numpy(),
            (mask_em & mask_EE).to_numpy(),

            (mask_me & mask_BB).to_numpy(),
            (mask_me & mask_BE).to_numpy(),
            (mask_me & mask_EB).to_numpy(),
            (mask_me & mask_EE).to_numpy(),

            (mask_mm & mask_BB).to_numpy(),
            (mask_mm & mask_BE).to_numpy(),
            (mask_mm & mask_EB).to_numpy(),
            (mask_mm & mask_EE).to_numpy()
        ], np.arange(0,16), 16)

        # this is to avoid cases were two 
        # leptons are not in the event
        lep_1_bin[lep_1_bin>6] = -1
        lep_2_bin[lep_2_bin>6] = -1
        center_value = self.trig_sf_map[lep_1_bin,lep_2_bin,trigg_bin,0]
        errors_value = self.trig_sf_map[lep_1_bin,lep_2_bin,trigg_bin,1]
        
        weights.add(
            'triggerSF', 
            center_value, 
            center_value + errors_value,
            center_value - errors_value
        )


    def process_shift(self, event, shift_name:str=''):
        _data_path = os.path.join(os.path.dirname(__file__), 'data/')
        dataset = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        selection = PackedSelection()
        weights = Weights(len(event), storeIndividual=True)
        
        histos = self.build_histos()
        
        if is_data:
            selection.add('lumimask', self._json[self._era](event.run, event.luminosityBlock))
            selection.add('triggers', trigger_rules(event, self._triggers, self._era))
        else:
            selection.add('lumimask', np.ones(len(event), dtype='bool'))
            selection.add('triggers', np.ones(len(event), dtype='bool'))
        
        # MET filters
        if is_data:
            selection.add(
                'metfilter',
                #event.Flag.METFilters &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter &
                event.Flag.goodVertices &
                event.Flag.eeBadScFilter &
                event.Flag.globalTightHalo2016Filter &
                event.Flag.BadChargedCandidateFilter & 
                event.Flag.BadPFMuonFilter
            )
        else:
            selection.add(
                'metfilter',
                #event.Flag.METFilters &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter & 
                event.Flag.goodVertices &
                event.Flag.eeBadScFilter &
                event.Flag.globalTightHalo2016Filter &
                event.Flag.BadChargedCandidateFilter & 
                event.Flag.BadPFMuonFilter
            )


        tight_lep, loose_lep, non_iso_leptons = build_leptons(
            event.Muon,
            event.Electron
        )
        
        had_taus = build_htaus(event.Tau, tight_lep)
        had_taus_loose = build_htaus_loose(event.Tau, tight_lep)
        had_taus_tight = build_htaus_tight(event.Tau, tight_lep)

        ntight_lep = ak.num(tight_lep)
        nloose_lep = ak.num(loose_lep)
        nhtaus_lep = ak.num(had_taus)
        nhtaus_lep_loose = ak.num(had_taus_loose)
        nhtaus_lep_tight = ak.num(had_taus_tight)

        lead_tau = ak.firsts(had_taus)
        lead_tau_loose = ak.firsts(had_taus_loose)
        lead_tau_tight = ak.firsts(had_taus_tight)

        tau_pt = lead_tau.pt
        taus_phi = lead_tau.phi
        taus_eta = lead_tau.eta
        tau_pt = lead_tau.pt
        tau_E = lead_tau.E
        lead_tau_loose = ak.firsts(had_taus_loose)
        tau_pt_loose = lead_tau_loose.pt
        taus_phi_loose = lead_tau_loose.phi
        taus_eta_loose = lead_tau_loose.eta
        tau_E_loose = lead_tau_loose.E

        lead_non_iso_lep = ak.firsts(non_iso_leptons)
        lead_lep_tight = ak.firsts(tight_lep)
        lead_lep_loose = ak.firsts(loose_lep)

        delta_R_non_iso_lep_loose_tau = lead_non_iso_lep.delta_r(lead_tau_loose)
        delta_R_non_iso_lep_vtight_tau = lead_lep_tight.delta_r(lead_tau_loose)
        delta_R_non_iso_lep_tight_tau = lead_lep_loose.delta_r(lead_tau_loose)

        deep_tau_e = lead_tau_loose.rawDeepTau2017v2p1VSe
        deep_tau_mu = lead_tau_loose.rawDeepTau2017v2p1VSmu
        deep_tau_jet = lead_tau_loose.rawDeepTau2017v2p1VSjet

        
        jets = event.Jet
        # overlap_leptons = ak.any(
        #     jets.metric_table(tight_lep) <= 0.4,
        #     axis=2
        # )
        # overlap_taus = ak.any(
        #     jets.metric_table(had_taus_loose) <= 0.4, #replaced vtight tau to loose tau
        #     axis=2
        # )
        
        # jet_mask = (
        #     ~overlap_leptons & 
        #     ~overlap_taus &
        #     (jets.pt>30.0) & 
        #     (np.abs(jets.eta) < 4.7) & 
        #     (jets.jetId >= 6) & # tight JetID 7(2016) and 6(2017/8)
        #     ((jets.puId >= 6) | (jets.puId == 3) | (jets.pt >= 50)) # medium puID https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL 3,7 for 16and 16APV; 6,7 for 17,18
        # )

        # jet_btag = (
        #         event.Jet.btagDeepFlavB > btag_id(
        #             self.btag_wp, 
        #             self._era + 'APV' if self._isAPV else self._era
        #         )
        # )
        
        good_jets, good_bjet = build_jets(jets, tight_lep, had_taus_loose, self.btag_wp, self._era, self._isAPV)
        # good_jets = jets[jet_mask] # removed ~jet_btag
        # good_bjet = jets[jet_btag & jet_mask & (np.abs(jets.eta)<2.4)]
        
    
        ngood_jets  = ak.num(good_jets)
        ngood_bjets = ak.num(good_bjet)
        
        event['ngood_bjets'] = ngood_bjets
        event['ngood_jets']  = ngood_jets
       
        # lepton quantities
        def z_lepton_pair(leptons):
            pair = ak.combinations(leptons, 2, axis=1, fields=['l1', 'l2'])
            mass = (pair.l1 + pair.l2).mass
            cand = ak.local_index(mass, axis=1) == ak.argmin(np.abs(mass - self.zmass), axis=1)

            extra_lepton = leptons[(
                ~ak.any(leptons.metric_table(pair[cand].l1) <= 0.01, axis=2) & 
                ~ak.any(leptons.metric_table(pair[cand].l2) <= 0.01, axis=2) )
            ]
            return pair[cand], extra_lepton, cand
        
        dilep, extra_lep, z_cand_mask = z_lepton_pair(tight_lep)
        
        lead_lep = ak.firsts(ak.where(dilep.l1.pt >  dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        subl_lep = ak.firsts(ak.where(dilep.l1.pt <= dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        
        
        dilep_p4 = (lead_lep + subl_lep)
        dilep_m  = dilep_p4.mass
        dilep_pt = dilep_p4.pt
        dilep_E = dilep_p4.t

        # high level observables
        p4_met = ak.zip(
            {
                "pt": event.MET.pt,
                "eta": ak.zeros_like(event.MET.pt),
                "phi": event.MET.phi,
                "mass": ak.zeros_like(event.MET.pt),
                "charge": ak.zeros_like(event.MET.pt),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        emu_met = ak.firsts(extra_lep, axis=1) + p4_met
        reco_met_pt = ak.where(ntight_lep==2, p4_met.pt, emu_met.pt)
        reco_met_phi = ak.where(ntight_lep==2, p4_met.phi, emu_met.phi)

    
        # this definition is not correct as it doesn't include the mass of the second Z
        dilep_et_ll = np.sqrt(dilep_pt**2 + dilep_m**2)
        dilep_et_nunu = np.sqrt(reco_met_pt**2 + self.zmass**2)
        dilep_mt_llnunu = ak.where(
                ntight_lep==3,
                # mT = m^2 + px^2 + py^2 = E^2 - pz^2
                np.sqrt((dilep_et_ll + dilep_et_nunu)**2 - ((dilep_p4.pvec + emu_met.pvec).pt)**2),
                np.sqrt((dilep_et_ll + dilep_et_nunu)**2 - ((dilep_p4.pvec +  p4_met.pvec).pt)**2)
        )
    
        dilep_dphi = lead_lep.delta_phi(subl_lep)
        dilep_deta = np.abs(lead_lep.eta - subl_lep.eta)
        dilep_dR   = lead_lep.delta_r(subl_lep)

        delta_R = ak.where(ntight_lep==2, dilep_p4.delta_r(lead_tau), dilep_p4.delta_r(lead_tau))
        dilep_dphi_met  = ak.where(ntight_lep==2, dilep_p4.delta_phi(p4_met), dilep_p4.delta_phi(emu_met))
        #scalar_balance = ak.where(ntight_lep==3, emu_met.pt/dilep_p4.pt, p4_met.pt/dilep_p4.pt)
        delta_tau_met_phi = ak.where(ntight_lep==2, lead_tau.delta_phi(p4_met), lead_tau.delta_phi(emu_met))
        dilep_dphi_tau = ak.where(ntight_lep==2, dilep_p4.delta_phi(lead_tau), dilep_p4.delta_phi(lead_tau))
        delta_tau_loose_met_phi = ak.where(ntight_lep==2, lead_tau_loose.delta_phi(p4_met), lead_tau_loose.delta_phi(emu_met))
        dilep_dphi_tau_loose = ak.where(ntight_lep==2, dilep_p4.delta_phi(lead_tau_loose), dilep_p4.delta_phi(lead_tau_loose))


        
        #Transverse WZ mass system 
        # Building 4 vector for tranverse mass calculation
        # .t is synonym for energy but there is a bug when we add option types of arrays of two leptons
        dilep_loose_tau_met_p4 = dilep_p4 + lead_tau_loose + p4_met
        mT_WZ_square = ((dilep_loose_tau_met_p4.t**2) - (dilep_loose_tau_met_p4.pz**2))
        mT_WZ = np.sqrt(np.maximum(0, mT_WZ_square))


        dilep_tau_loose_met_hadron_mt = np.sqrt((transverse_energy(lead_lep) + transverse_energy(subl_lep) + transverse_energy(lead_tau_loose) + p4_met.pt) ** 2 - dilep_loose_tau_met_p4.pt**2)

        inv_m_WZ = (dilep_loose_tau_met_p4).mass


        #tranverse W mass
        tau_loose_met_p4 = lead_tau_loose + p4_met
        mT_W_square = ((tau_loose_met_p4.t)**2 - (tau_loose_met_p4.pz**2))
        mT_W = np.sqrt(np.maximum(0, mT_W_square))

        #HT, scalar sum of jet pt, and HTl, HT + lepton pt
        HT = ak.sum(good_jets.pt, axis=1)
        HTl = HT + lead_lep.pt + subl_lep.pt + lead_tau_loose.pt

        # ST, scalar sum of all object pts
        ST = HTl + p4_met.pt

        
        # 2jet and vbs related variables
        lead_jet = ak.firsts(good_jets)
        subl_jet = ak.firsts(good_jets[lead_jet.delta_r(good_jets)>0.01])
        third_jet = ak.firsts(good_jets[(lead_jet.delta_r(good_jets)>0.01) & (subl_jet.delta_r(good_jets)>0.01)])
        delta_R_jet_dilep = ak.where(ntight_lep==2, dilep_p4.delta_r(lead_jet), dilep_p4.delta_r(lead_jet))
        delta_R_jet_tau = ak.where(ntight_lep==2, lead_tau.delta_r(lead_jet), lead_tau.delta_r(lead_jet))
        dphi_jet_met = ak.where(ntight_lep==2, lead_jet.delta_phi(p4_met), lead_jet.delta_phi(emu_met))
        
        dijet_mass = (lead_jet + subl_jet).mass
        dijet_deta = np.abs(lead_jet.eta - subl_jet.eta)
        event['dijet_mass'] = dijet_mass
        event['dijet_deta'] = dijet_deta 
        
        min_dphi_met_j = ak.min(np.abs(
            ak.where(
                ntight_lep==3, 
                good_jets.delta_phi(emu_met), 
                good_jets.delta_phi(p4_met)
            )
        ), axis=1)

        event['min_dphi_met_j'] = min_dphi_met_j
        
        # define basic selection
        selection.add(
            "require-ossf",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        
        selection.add(
            "require-osof",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
           ak.fill_none(np.abs(lead_lep.pdgId) != np.abs(subl_lep.pdgId), False)
        )
        
        selection.add(
            "require-2lep",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )

        selection.add(
            "require-3lep",
            (ntight_lep==3) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        
        selection.add(
            "require-4lep",
            (ntight_lep>=2) & (nloose_lep + ntight_lep)==4 &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )

        selection.add('met_pt', ak.fill_none((reco_met_pt > 30), False))
        selection.add('low_met_pt', ak.fill_none((reco_met_pt < 20) & (reco_met_pt > 0), False))
        selection.add('dilep_m'   , ak.fill_none(np.abs(dilep_m - self.zmass) < 15, False))
        selection.add('dilep_pt', ak.fill_none(dilep_pt > 30, False))
        selection.add("dilep_dphi_met", ak.fill_none(np.abs(dilep_dphi_met)>1.0, False))
        selection.add("dilep_dphi_tau", ak.fill_none(np.abs(dilep_dphi_tau)>1.0, False))
        selection.add("delta_tau_met_phi", ak.fill_none(np.abs(delta_tau_met_phi)>1.0, False))
        selection.add("dilep_dphi_tau_loose", ak.fill_none(np.abs(dilep_dphi_tau_loose)>1.0, False))
        selection.add("delta_tau_loose_met_phi", ak.fill_none(np.abs(delta_tau_loose_met_phi)>1.0, False))
        selection.add(
            "min_dphi_met_j",
            ak.where(
                ngood_jets <= 1, 
                ak.fill_none(np.abs(min_dphi_met_j)>0.25, False), 
                ak.fill_none(np.abs(min_dphi_met_j)>0.5, False), 
            )
        )
        # jet demography
        selection.add('0njets' , ngood_jets  == 0 )
        selection.add('1njets' , ngood_jets  <= 1 )
        selection.add('1njets_only' , ngood_jets  == 1 )
        selection.add('2njets' , ngood_jets  <= 2 )
        selection.add('3njets' , ngood_jets  >= 3 )
        selection.add('0nbjets', ngood_bjets == 0 )
        selection.add('nbjets', ngood_bjets >= 1 )
        selection.add('1nhtaus', nhtaus_lep  == 1 )
        selection.add('1nhtaus_loose', nhtaus_lep_loose  == 1 )
        selection.add('1nhtaus_tight', nhtaus_lep_tight  == 1 )
        
        selection.add('dijet_deta', ak.fill_none(dijet_deta > 2.5, False))
        selection.add('dijet_mass_400' , ak.fill_none(dijet_mass >  400, False))
        selection.add('dijet_mass_800' , ak.fill_none(dijet_mass >  800, False))

        # Define all variables for the BDT
        event['met_pt'  ] = ak.fill_none(reco_met_pt,-99)
        event['met_phi'  ] = ak.fill_none(reco_met_phi,-99)
        event['mT_W'  ] = ak.fill_none(mT_W,-99)
        event['mT_WZ'  ] = ak.fill_none(mT_WZ,-99)
        event['inv_m_WZ'  ] = ak.fill_none(inv_m_WZ,-99)
        event['dilep_tau_loose_met_hadron_mt'  ] = ak.fill_none(dilep_tau_loose_met_hadron_mt,-99)
        event['met_phi' ] = ak.fill_none(reco_met_phi,-99)
        event['dilep_mt_llnunu'] = ak.fill_none(dilep_mt_llnunu,-99)
        event['dilep_m'] = ak.fill_none(dilep_m,-99)
        event['dilep_pt'] = ak.fill_none(dilep_pt,-99)
        event['HTl'] = ak.fill_none(HTl,-99)
        event['ST'] = ak.fill_none(ST,-99)
        event['dilep_dphi'] = ak.fill_none(dilep_dphi,-99)
        event['njets'   ] = ak.fill_none(ngood_jets,-99)
        event['nbjets'   ] = ak.fill_none(ngood_bjets,-99)
        event['ntaus_vtight'   ] = ak.fill_none(nhtaus_lep,-99)
        event['ntaus_tight'   ] = ak.fill_none(nhtaus_lep_tight,-99)
        event['ntaus_loose'   ] = ak.fill_none(nhtaus_lep_loose,-99)
        event['dphi_met_ll'] = ak.fill_none(dilep_dphi_met,-99)
        event['dilep_dphi_tau'] = ak.fill_none(dilep_dphi_tau,-99)
        event['dijet_mass'] = ak.fill_none(dijet_mass,-99)
        event['dijet_deta'] = ak.fill_none(dijet_deta,-99)
        event['min_dphi_met_j'] = ak.fill_none(min_dphi_met_j,-99)
        event['tau_pt'] = ak.fill_none(tau_pt,-99)
        event['taus_phi'] = ak.fill_none(taus_phi,-99)
        event['taus_eta'] = ak.fill_none(taus_eta,-99)
        event['delta_R'] = ak.fill_none(delta_R,-99)
        event['dilep_dR'] = ak.fill_none(dilep_dR,-99)
        event['dilep_deta'] = ak.fill_none(dilep_deta,-99)
        event['delta_tau_met_phi'] = ak.fill_none(delta_tau_met_phi,-99)
        event['tau_pt_loose'] = ak.fill_none(tau_pt_loose,-99)
        event['taus_phi_loose'] = ak.fill_none(taus_phi_loose,-99)
        event['taus_eta_loose'] = ak.fill_none(taus_eta_loose,-99)
        event['leading_lep_pt'  ] = ak.fill_none(lead_lep.pt,-99)
        event['leading_lep_eta' ] = ak.fill_none(lead_lep.eta,-99)
        event['leading_lep_phi' ] = ak.fill_none(lead_lep.phi,-99)
        event['trailing_lep_pt' ] = ak.fill_none(subl_lep.pt,-99)
        event['trailing_lep_eta'] = ak.fill_none(subl_lep.eta,-99)
        event['trailing_lep_phi'] = ak.fill_none(subl_lep.phi,-99)       
        event['lead_jet_pt'  ] = ak.fill_none(lead_jet.pt,-99)
        event['lead_jet_eta' ] = ak.fill_none(lead_jet.eta,-99)
        event['lead_jet_phi' ] = ak.fill_none(lead_jet.phi,-99)
        event['delta_R_jet_tau'] = ak.fill_none(delta_R_jet_tau,-99)
        event['delta_R_jet_dilep'] = ak.fill_none(delta_R_jet_dilep,-99)
        event['dphi_jet_met'] = ak.fill_none(dphi_jet_met,-99)
        event['deep_tau_e'] = ak.fill_none(deep_tau_e,-99)
        event['deep_tau_mu'] = ak.fill_none(deep_tau_mu,-99)
        event['deep_tau_jet'] = ak.fill_none(deep_tau_jet,-99)
        event['delta_R_non_iso_lep_loose_tau'] = ak.fill_none(delta_R_non_iso_lep_loose_tau,-99)
        event['delta_R_non_iso_lep_vtight_tau'] = ak.fill_none(delta_R_non_iso_lep_vtight_tau,-99)
        event['delta_R_non_iso_lep_tight_tau'] = ak.fill_none(delta_R_non_iso_lep_tight_tau,-99)


        # Now adding weights
        _ones = np.ones(len(weights.weight()))
        if not is_data:
            weights.add('genweight', event.genWeight)
            # self._btag.append_btag_sf(jets, weights)
            self._jpSF.append_jetPU_sf(good_jets, weights)
            self._purw.append_pileup_weight(weights, event.Pileup.nTrueInt) # fix: https://github.com/9GaoHong/SMQawa_update/commit/d6cdebda4856593162c03365eb9d9a91ceb1a185
            self._tauID.append_tauID_sf(had_taus, weights)
            self._add_trigger_sf(weights, lead_lep, subl_lep)
            
            weights.add (
                    'LeptonSF', 
                    lead_lep.SF*subl_lep.SF, 
                    lead_lep.SF_up*subl_lep.SF_up, 
                    lead_lep.SF_down*subl_lep.SF_down
            )
            
            if self.ewk_process_name:
                self.ewk_corr.get_weight(
                        event.GenPart,
                        event.Generator.x1,
                        event.Generator.x2,
                        weights
                )
            else:
                weights.add("kEW", _ones, _ones, _ones)
            
            if "PSWeight" in event.fields:
                theory_ps_weight(weights, event.PSWeight)
            else:
                theory_ps_weight(weights, None)
            
            if "LHEPdfWeight" in event.fields:
                theory_pdf_weight(weights, event.LHEPdfWeight)
            else:
                theory_pdf_weight(weights, None)

            if ('LHEScaleWeight' in event.fields) and (len(event.LHEScaleWeight[0]) > 0):
                if len(event.LHEScaleWeight[0]) == 9:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 7])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 5])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 8])
                elif len(event.LHEScaleWeight[0]) == 8:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 6])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 4])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 7])
                elif len(event.LHEScaleWeight[0]) == 18:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 2], event.LHEScaleWeight[:, 14])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 6], event.LHEScaleWeight[:, 10])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 16])
                else:
                    print("WARNING: QCD scale variation type not recongnised ... ")
                
            if 'LHEReweightingWeight' in event.fields and 'aQGC' in dataset:
                for i in range(1057):
                    weights.add(f"eft_{self._eftnames[i]}", event.LHEReweightingWeight[:, i])
            
            # 2017 Prefiring correction weight
            if 'L1PreFiringWeight' in event.fields:
                weights.add("prefiring_weight", event.L1PreFiringWeight.Nom, event.L1PreFiringWeight.Dn, event.L1PreFiringWeight.Up)
        else:
            # If systematic variations are needed, they must be manually inserted here to give different DD estimates; they should be picked up later for histos.
            weights.add("datadriven_DDDYNominal", _ones, self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, systematic="nominal"), self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, systematic="nominal"))  #added nominal value twice to avoid getting 1/up for the nominaldown
            weights.add("datadriven_DDDY",_ones, self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, "DDDYUp"), self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, "DDDYDown"))
        # selections (delta_tau_met_phi cut is removed from SR)

        common_sel = ['triggers', 'lumimask', 'metfilter']
        channels = {
            "inc-SR0": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', '0njets', '1nhtaus', 'met_pt', 'dilep_pt'
        ],
            "inc-SR1": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', '1njets_only', '1nhtaus' ,'met_pt', 'dilep_pt'
        ],
        #     "inc-SR1l": common_sel + [
        #     'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', '1njets_only', '1nhtaus' ,'met_pt', 'dilep_pt', '0nbjets'
        # ],
        #     "inc-SR1b": common_sel + [
        #     'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', '1njets_only', '1nhtaus' ,'met_pt', 'dilep_pt', '~0nbjets'
        # ],
            "inc-SR01": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', '1njets', '1nhtaus', 'met_pt', 'dilep_pt'
        ],
            "inc-DY0": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'met_pt','dilep_pt', '0njets',  '~1nhtaus', '1nhtaus_loose', '~1nhtaus_tight'
        ],
            "inc-DY1": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'met_pt','dilep_pt', '1njets_only', '~1nhtaus', '1nhtaus_loose', '~1nhtaus_tight'
        ],
            "inc-EM0": common_sel + [
            'require-osof', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', '1nhtaus', 'met_pt', '0njets'
        ],
            "inc-EM1": common_sel + [
            'require-osof', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', '1nhtaus', 'met_pt', '1njets_only'
        ],

            "inc-B0": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', '1nhtaus_loose', 'met_pt', '0njets', '~1nhtaus_tight', '~1nhtaus'
        ],

            "inc-C0": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', 'low_met_pt', '1nhtaus_loose', '0njets', '~1nhtaus_tight', '~1nhtaus'
        ],

            "inc-D0": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', 'low_met_pt', '1nhtaus', '0njets'
        ],

            "inc-B1": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', '1nhtaus_loose', 'met_pt', '1njets_only', '~1nhtaus_tight', '~1nhtaus'
        ],

            "inc-C1": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', 'low_met_pt', '1nhtaus_loose', '1njets_only', '~1nhtaus_tight', '~1nhtaus'
        ],

            "inc-D1": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', 'low_met_pt', '1nhtaus', '1njets_only'
        ],

            "inc-B01": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', '1nhtaus_loose', 'met_pt', '1njets', '~1nhtaus_tight', '~1nhtaus'
        ],
        }
            
        def _format_variable(variable, cut):
            if cut is None:
                vv = ak.to_numpy(ak.fill_none(variable, np.nan))
                if np.isnan(np.any(vv)):
                    print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable, np.nan))
            else:
                vv = ak.to_numpy(ak.fill_none(variable[cut], np.nan))
                if np.isnan(np.any(vv)):
                    print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable[cut], np.nan))

        def collection_printer(collection):
            longest_field = max([len(field) for field in collection.fields])
            for field in collection.fields:
                print(f"\t{field:<{longest_field}}={getattr(collection, field)}")
        
        def _histogram_filler(ch, syst, var, _weight=None):
            sel_ = channels[ch]
            sel_args_ = {
                s.replace('~',''): (False if '~' in s else True) for s in sel_ if var not in s
            }
            cut =  selection.require(**sel_args_)

            systname = 'nominal' if syst is None else syst
            
            if _weight is None: 
                if syst in weights.variations:
                    weight = weights.weight(modifier=syst)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * _weight[cut]
            
            vv = ak.to_numpy(ak.fill_none(weight, np.nan))
            if np.isnan(np.any(vv)):
                print(f" - {syst} weight contains invalid values:", vv[np.isnan(vv)], vv[np.isinf(vv)])

            # if ch in ['inc-SR1', 'inc-DY1']:
            #     if var in ["met_pt", "mT_WZ", "lead_jet_pt", "dilep_loose_tau_pt", "dilep_loose_tau_met_dphi", "met_phi", "lead_jet_phi", "dilep_loose_tau_phi"] :
            #         if systname in ['nominal', 'JESUp', 'JESDown']:
            #             print(ch, var, systname, _format_variable(event[var], cut)[:2], event.event[cut][:2])
            #             if ch=='inc-SR1' and var=="mT_WZ" :
            #                 collection_printer(event.Tau[:2])


            histos[var].fill(
                **{
                    "channel": ch, 
                    "systematic": systname, 
                    var: _format_variable(event[var], cut), 
                    "weight": ak.nan_to_num(weight,nan=1.0, posinf=1.0, neginf=1.0)
                }
            )

        def _histogram_filler2D(ch, syst, var1, var2, _weight=None):
            sel_ = channels[ch]
            sel_args_ = {
                s.replace('~',''): (False if '~' in s else True) for s in sel_ if var1 not in s and var2 not in s
            }
            cut =  selection.require(**sel_args_)

            systname = 'nominal' if syst is None else syst
            
            if _weight is None: 
                if syst in weights.variations:
                    weight = weights.weight(modifier=syst)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * _weight[cut]
            
            vv = ak.to_numpy(ak.fill_none(weight, np.nan))
            if np.isnan(np.any(vv)):
                print(f" - {syst} weight contains invalid values:", vv[np.isnan(vv)], vv[np.isinf(vv)])

            histos[var1+"_2D_"+var2].fill(
                **{
                    "channel": ch, 
                    "systematic": systname, 
                    var1: _format_variable(event[var1], cut),
                    var2: _format_variable(event[var2], cut), 
                    "weight": ak.nan_to_num(weight,nan=1.0, posinf=1.0, neginf=1.0)
                }
            )
        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]
            
        for ch in channels:
            for sys in systematics:
                _histogram_filler(ch, sys, 'leading_lep_pt')
                _histogram_filler(ch, sys, 'leading_lep_phi')
                _histogram_filler(ch, sys, 'leading_lep_eta')
                _histogram_filler(ch, sys, 'trailing_lep_pt')
                _histogram_filler(ch, sys, 'trailing_lep_phi')
                _histogram_filler(ch, sys, 'trailing_lep_eta')
                _histogram_filler(ch, sys, 'met_pt')
                _histogram_filler(ch, sys, 'met_phi')
                _histogram_filler(ch, sys, 'tau_pt')
                _histogram_filler(ch, sys, 'taus_phi')
                _histogram_filler(ch, sys, 'taus_eta')
                _histogram_filler(ch, sys, 'tau_pt_loose')
                _histogram_filler(ch, sys, 'taus_phi_loose')
                _histogram_filler(ch, sys, 'taus_eta_loose')
                _histogram_filler(ch, sys, 'lead_jet_pt')
                _histogram_filler(ch, sys, 'lead_jet_phi')
                _histogram_filler(ch, sys, 'lead_jet_eta')
                _histogram_filler(ch, sys, 'njets')
                _histogram_filler(ch, sys, 'nbjets')
                _histogram_filler(ch, sys, 'ntaus_loose')
                _histogram_filler(ch, sys, 'ntaus_tight')
                _histogram_filler(ch, sys, 'ntaus_vtight')
                _histogram_filler(ch, sys, 'dilep_pt')
                _histogram_filler(ch, sys, 'dilep_dphi')
                _histogram_filler(ch, sys, 'dilep_deta')
                _histogram_filler(ch, sys, 'dilep_m')
                _histogram_filler(ch, sys, 'dilep_dR')
                _histogram_filler(ch, sys, 'delta_R')
                _histogram_filler(ch, sys, 'delta_R_jet_tau')
                _histogram_filler(ch, sys, 'delta_R_jet_dilep')
                _histogram_filler(ch, sys, 'dphi_met_ll')
                _histogram_filler(ch, sys, 'dilep_dphi_tau')
                _histogram_filler(ch, sys, 'dphi_jet_met')
                _histogram_filler(ch, sys, 'delta_tau_met_phi')
                _histogram_filler(ch, sys, 'mT_W')
                _histogram_filler(ch, sys, 'mT_WZ')
                _histogram_filler(ch, sys, 'inv_m_WZ')
                _histogram_filler(ch, sys, 'dilep_tau_loose_met_hadron_mt')
                _histogram_filler(ch, sys, 'dilep_mt_llnunu')
                _histogram_filler(ch, sys, 'HTl')
                _histogram_filler(ch, sys, 'ST')
                _histogram_filler(ch, sys, 'deep_tau_e')
                _histogram_filler(ch, sys, 'deep_tau_mu')
                _histogram_filler(ch, sys, 'deep_tau_jet')
                _histogram_filler(ch, sys, 'delta_R_non_iso_lep_loose_tau')
                _histogram_filler(ch, sys, 'delta_R_non_iso_lep_tight_tau')
                _histogram_filler(ch, sys, 'delta_R_non_iso_lep_vtight_tau')

        return {dataset: histos}
        
    def process(self, event: processor.LazyDataFrame):
        dataset_name = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        

        # JES/JER corrections
        rho = event.fixedGridRhoFastjetAll
        cache = event.caches[0]

        
        raw_met = event.RawMET
        met_to_correct = event.MET
       
        jets = self._jmeu.corrected_jets_L123_JER(event.Jet, event.fixedGridRhoFastjetAll, event.caches[0])
        jets_to_correct_met = self._jmeu.corrected_jets_L123_noJER(event.Jet, event.fixedGridRhoFastjetAll, event.caches[0])
        met = self._jmeu.corrected_met(met_to_correct, jets, event.fixedGridRhoFastjetAll, event.caches[0]) # we are adding fully smeared L123 jets

        event = ak.with_field(event, event.Jet, 'OrigJet')
        event = ak.with_field(event, event.MET, 'OrigMET')
        event = ak.with_field(event, jets, 'Jet')
        event = ak.with_field(event, jets_to_correct_met, 'JetforMET')
        event = ak.with_field(event, met, 'MET')
        

        run = event.run 
        npv = event.PV.npvs
        
        met = met_phi_xy_correction(
            event.MET, run, npv, 
            is_mc=not is_data, 
            era=self._era
        )
        event = ak.with_field(event, met, 'MET')

    
        if is_data:
            
            # Apply rochester_correction
            muon = event.Muon 
            muon_pt,muon_pt_roccorUp,muon_pt_roccorDown=rochester_correction(is_data).apply_rochester_correction (muon)
            muon['pt'] = muon_pt
            event = ak.with_field(event, muon, 'Muon')

            return self.process_shift(event, None)

        
        # Adding scale factors to Muon and Electron fields
        muon = event.Muon 
        electron = event.Electron
        muonSF_nom, muonSF_up, muonSF_down = self._leSF.muonSF(muon)
        elecSF_nom, elecSF_up, elecSF_down = self._leSF.electronSF(electron)
        
        muon['SF'] = muonSF_nom
        muon['SF_up'] = muonSF_up
        muon['SF_down'] = muonSF_down

        electron['SF'] = elecSF_nom
        electron['SF_up'] = elecSF_up
        electron['SF_down'] = elecSF_down

        event = ak.with_field(event, muon, 'Muon')
        event = ak.with_field(event, electron, 'Electron')

        
        # Apply rochester_correction
        muon=event.Muon
        muonEnUp=event.Muon
        muonEnDown=event.Muon
        muon_pt,muon_pt_roccorUp,muon_pt_roccorDown=rochester_correction(is_data).apply_rochester_correction (muon)
        
        muon['pt'] = muon_pt
        muonEnUp['pt'] = muon_pt_roccorUp
        muonEnDown['pt'] = muon_pt_roccorDown 
        event = ak.with_field(event, muon, 'Muon')
        
        # Electron corrections
        electronEnUp=event.Electron
        electronEnDown=event.Electron

        electronEnUp['pt'] = event.Electron['pt'] + event.Electron.energyErr/np.cosh(event.Electron.eta)
        electronEnDown['pt'] = event.Electron['pt'] - event.Electron.energyErr/np.cosh(event.Electron.eta)  


        #Tau corrections
        # if not is_data :
        tau = event.Tau[(event.Tau.decayMode != 5) & (event.Tau.decayMode != 6)]
        tauEnUp = event.Tau[(event.Tau.decayMode != 5) & (event.Tau.decayMode != 6)]
        tauEnDown = event.Tau[(event.Tau.decayMode != 5) & (event.Tau.decayMode != 6)]
        tau_pt,tau_pt_EnUp,tau_pt_EnDown,tau_mass,tau_mass_EnUp,tau_mass_EnDown=self._tauID.tau_energy_scale_correction(tau)

        tau['pt'] = tau_pt
        tau['mass'] = tau_mass
        tauEnUp['pt'] = tau_pt_EnUp
        tauEnUp['mass'] = tau_mass_EnUp
        tauEnDown['pt'] = tau_pt_EnDown
        tauEnDown['mass'] = tau_mass_EnDown
        event = ak.with_field(event, tau, 'Tau')

    
        # define all the shifts
        shifts = [
            # Jets
            ({"Jet": jets                             , "MET": met                               }, None                  ),
            ({"Jet": jets.JES_Total.up                , "MET": met.JES_Total.up                  }, "JESUp"               ),
            ({"Jet": jets.JES_Total.down              , "MET": met.JES_Total.down                }, "JESDown"             ),
            ({"Jet": jets.JES_Absolute.up             , "MET": met.JES_Absolute.up               }, "JES_AbsoluteUp"      ),
            ({"Jet": jets.JES_Absolute.down           , "MET": met.JES_Absolute.down             }, "JES_AbsoluteDown"    ),
            ({"Jet": jets.JES_BBEC1.up                , "MET": met.JES_BBEC1.up                  }, "JES_BBEC1Up"         ),
            ({"Jet": jets.JES_BBEC1.down              , "MET": met.JES_BBEC1.down                }, "JES_BBEC1Down"       ),
            ({"Jet": jets.JES_EC2.up                  , "MET": met.JES_EC2.up                    }, "JES_EC2Up"           ),
            ({"Jet": jets.JES_EC2.down                , "MET": met.JES_EC2.down                  }, "JES_EC2Down"         ),
            ({"Jet": jets.JES_FlavorQCD.up            , "MET": met.JES_FlavorQCD.up              }, "JES_FlavorQCDUp"     ),
            ({"Jet": jets.JES_FlavorQCD.down          , "MET": met.JES_FlavorQCD.down            }, "JES_FlavorQCDDown"   ),
            ({"Jet": jets.JES_HF.up                   , "MET": met.JES_HF.up                     }, "JES_HFUp"            ),
            ({"Jet": jets.JES_HF.down                 , "MET": met.JES_HF.down                   }, "JES_HFDown"          ),
            ({"Jet": jets.JES_RelativeBal.up          , "MET": met.JES_RelativeBal.up            }, "JES_RelativeBalUp"   ),
            ({"Jet": jets.JES_RelativeBal.down        , "MET": met.JES_RelativeBal.down          }, "JES_RelativeBalDown" ),
            ({"Jet": jets.JER.up                      , "MET": met.JER.up                        }, "JERUp"               ),
            ({"Jet": jets.JER.down                    , "MET": met.JER.down                      }, "JERDown"             ),
            ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.up      }, "UESUp"               ),
            ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.down    }, "UESDown"             ), 
              ##year dependent systematics
            ({"Jet": getattr(jets,f'JES_BBEC1_{self._era}').up     , "MET": getattr(met,f'JES_BBEC1_{self._era}').up      }, f"JES_BBEC1{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_BBEC1_{self._era}').down   , "MET": getattr(met,f'JES_BBEC1_{self._era}').down    }, f"JES_BBEC1{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_Absolute_{self._era}').up  , "MET": getattr(met,f'JES_Absolute_{self._era}').up   }, f"JES_Absolute{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_Absolute_{self._era}').down, "MET": getattr(met,f'JES_Absolute_{self._era}').down }, f"JES_Absolute{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_EC2_{self._era}').up       , "MET": getattr(met,f'JES_EC2_{self._era}').up        }, f"JES_EC2{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_EC2_{self._era}').down     , "MET": getattr(met,f'JES_EC2_{self._era}').down      }, f"JES_EC2{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_HF_{self._era}').up        , "MET": getattr(met,f'JES_HF_{self._era}').up         }, f"JES_HF{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_HF_{self._era}').down      , "MET": getattr(met,f'JES_HF_{self._era}').down       }, f"JES_HF{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_RelativeSample_{self._era}').up  , "MET": getattr(met,f'JES_RelativeSample_{self._era}').up   }, f"JES_RelativeSample{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_RelativeSample_{self._era}').down, "MET": getattr(met,f'JES_RelativeSample_{self._era}').down }, f"JES_RelativeSample{self._era}Down"),

            
            # Electrons + MET shift (FIXME: shift to be added)
            ({"Electron": electronEnUp  }, "ElectronEnUp"  ),
            ({"Electron": electronEnDown}, "ElectronEnDown"),
            # Muon + MET shifts
            ({"Muon": muonEnUp  }, "MuonRocUp"),
            ({"Muon": muonEnDown}, "MuonRocDown"),
            # Tau + MET shifts
            ({"Tau": tauEnUp  }, "TauEnUp"),
            ({"Tau": tauEnDown}, "TauEnDown"),

        ]

        if (self._era == '2018') and (not is_data):
            hem_jets, hem_met = apply_hem_uncertainty(
                event.Jet,
                event.MET
            )
            shifts.append(({"Jet": hem_jets, "MET": hem_met}, "HEMDown"))
            shifts.append(({"Jet": event.Jet, "MET": event.MET}, "HEMUp"))

        shifts = [
            self.process_shift(
                update_collection(event, collections), 
                name
            ) for collections, name in shifts
        ]
        return processor.accumulate(shifts)
    

    
    def postprocess(self, accumulator):
        return accumulator

