import os

import awkward as ak
import numpy as np
import scipy.interpolate as interp
from scipy import stats as st
import uproot
import pickle
import hist
import yaml
import copy
import os
import re
import gzip
from coffea import processor
from coffea import nanoevents
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import nanoaod
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask

from qawa.roccor import rochester_correction
# from qawa.applyGNN_old import applyGNN
from qawa.leptonsSF import LeptonScaleFactors
from qawa.jetPU import jetPUScaleFactors
from qawa.tauSF import tauIDScaleFactors
from qawa.btag import BTVCorrector, btag_id
from qawa.jme import JMEUncertainty, update_collection
from qawa.gen_match import delta_r2, find_best_match
# from qawa.ddr_SR import dataDrivenDYRatio
from qawa.common import pileup_weights, ewk_corrector, met_phi_xy_correction, theory_ps_weight, theory_pdf_weight, trigger_rules

def build_leptons(muons, electrons):
    tight_muons_mask = (
        (muons.pt             >  20. ) &
        (np.abs(muons.eta)    <  2.4 ) &
        (np.abs(muons.dxy)    <  0.02) &
        (np.abs(muons.dz )    <  0.1 ) &
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
    SCeta = np.abs(electrons.eta + electrons.deltaEtaSC)
    tight_electrons_mask = (
        (electrons.pt           > 20.) &
        (((SCeta  < 2.5) &
        (SCeta  > 1.5660)) | 
        (SCeta  < 1.4442)) &
        electrons.mvaFall17V2Iso_WP90 & 
        electrons.mvaFall17V2Iso_WPL
    )
    tight_electrons = electrons[tight_electrons_mask]
    loose_electrons = electrons[
        ~tight_electrons_mask &
        (electrons.pt           > 10. ) &
        (np.abs(electrons.eta)  < 2.5) &
        electrons.mvaFall17V2Iso_WPL
    ]

    tight_leptons = ak.with_name(ak.concatenate([tight_muons, tight_electrons], axis=1), 'PtEtaPhiMCandidate')
    nloose = ak.num(loose_muons) + ak.num(loose_electrons)
    
    tight_sorted_index = ak.argsort(tight_leptons.pt,ascending=False)
    tight_leptons = tight_leptons[tight_sorted_index]

    return tight_leptons, nloose 

def build_htaus(tau, lepton):
    print("build_htaus function using probably too-loose selections, should be re-evaluated for rejecting taus: calibration available for VTight, Tight, Medium, Loose VSjet scores (64, 32, 16, 8) but only Loose and Tight VSe scores (1==VVVLoose), which should be paired appropriately in the TauSF code")
    base = (
        (tau.pt         > 20 ) & 
        (np.abs(tau.eta)< 2.3 ) & 
        (tau.decayMode != 5   ) & 
        (tau.decayMode != 6   ) &
        (tau.idDeepTau2017v2p1VSe >= 2) &
        (tau.idDeepTau2017v2p1VSmu >= 1) &
        (tau.idDeepTau2017v2p1VSjet >= 16)
    )
    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )
    return tau[base & ~overlap_leptons]

def apply_hem_uncertainty(jets, met, overlap_leptons=None):
    if overlap_leptons is None:
        lepton_mask = ak.ones_like(jets.pt, dtype=np.bool_)
    else:
        lepton_mask = ~overlap_leptons

    phi_mask = (
        (jets.phi > -1.57) &
        (jets.phi < -0.87)
    )
    tight_mask = (
        lepton_mask &
        (jets.pt > 15.0) &
        (jets.jetId >= 6) &
        phi_mask
    )
    mask_20 = tight_mask & (jets.eta > -2.5) & (jets.eta < -1.3)
    mask_35 = tight_mask & (jets.eta > -3.0) & (jets.eta < -2.5)

    scale = ak.ones_like(jets.pt)
    scale = ak.where(mask_20, 0.80, scale)
    scale = ak.where(mask_35, 0.65, scale)

    scaled_jets = ak.with_field(jets, jets.pt * scale, 'pt')
    scaled_jets = ak.with_field(scaled_jets, jets.mass * scale, 'mass')

    delta_px = ak.sum((jets.pt - scaled_jets.pt) * np.cos(jets.phi), axis=1, mask_identity=False)
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

class zzinc_processor(processor.ProcessorABC):
    # EWK corrections process has to be define before hand, it has to change when we move to dask
    def __init__(self, era: str ='2018', isDY=False, dd='SR',dump_gnn_array=False, ewk_process_name=None, run_period: str = ''): 

        self._era = era
        self._isDY = isDY
        self._ddtype = dd
        if 'APV' in self._era:
            self._isAPV = True
            self._era = re.findall(r'\d+', self._era)[0] 
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
            else:
                print('error')
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
            else:
                print('error')
        
        self.btag_wp = 'M'
        self.jetPU_wp = 'M'
        self.tauIDvsjet_wp = 'Medium'
        self.tauIDvse_wp = 'VVLoose'
        self.tauIDvsmu_wp = 'VLoose'
        self.zmass = 91.1873 # GeV 
        self._btag = BTVCorrector(era=self._era, wp=self.btag_wp, isAPV=self._isAPV)
        self._jmeu = JMEUncertainty(jec_tag, jer_tag, era=self._era, is_mc=(len(run_period)==0))
        self._purw = pileup_weights(era=self._era)
        self._leSF = LeptonScaleFactors(era=self._era, isAPV=self._isAPV)
        self._jpSF = jetPUScaleFactors(era=self._era, wp=self.jetPU_wp, isAPV=self._isAPV)
        self._tauID= tauIDScaleFactors(era=self._era, vsjet_wp=self.tauIDvsjet_wp,vse_wp=self.tauIDvse_wp, vsmu_wp=self.tauIDvsmu_wp, isAPV=self._isAPV)
        
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
        self.dump_gnn_array  = dump_gnn_array
        
        with open(f'{_data_path}/GNNmodel/gnn_flattening_fnc_{era}.pkl', 'rb') as _fn:
            self.gnn_flat_fnc = pickle.load(_fn)
        _gnn_flat_x = np.asarray(self.gnn_flat_fnc.x, dtype=float)
        _gnn_flat_y = np.asarray(self.gnn_flat_fnc.y, dtype=float)
        self._gnn_score_min = float(np.nanmin(_gnn_flat_x)) if _gnn_flat_x.size else 0.0
        self._gnn_score_max = float(np.nanmax(_gnn_flat_x)) if _gnn_flat_x.size else 1.0
        self._gnn_flat_min = float(max(0.0, np.nanmin(_gnn_flat_y))) if _gnn_flat_y.size else 0.0
        self._gnn_flat_max = float(min(1.0, np.nanmax(_gnn_flat_y))) if _gnn_flat_y.size else 1.0

        self.ewk_process_name = ewk_process_name
        if self.ewk_process_name is not None:
            self.ewk_corr = ewk_corrector(process=ewk_process_name)

        self.build_histos = lambda: {
            'dilep_mt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="dilep_mt", label=r"$M_{T}^{\ell\ell}$ (GeV)",flow=True),
                hist.storage.Weight()
            ), 
	        'dilep_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="dilep_pt", label=r"$p_{T}^{\ell\ell}$ (GeV)"),
                hist.storage.Weight()
            ), 
	        'dilep_m': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 120, name="dilep_m", label=r"$M_{\ell\ell}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'met_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="met_pt", label=r"$p_{T}^{miss}$ (GeV)",flow=True),
                hist.storage.Weight()
            ),
            # 'met_sigma': hist.Hist(
            #     hist.axis.StrCategory([], name="channel"   , growth=True),
            #     hist.axis.StrCategory([], name="systematic", growth=True), 
            #     hist.axis.Regular(50, 0, 100, name="met_sigma", label=r"$p_{T}^{miss} sigma$ (GeV)"),
            #     hist.storage.Weight()
            # ),
            # 'met_uncertainty': hist.Hist(
            #     hist.axis.StrCategory([], name="channel"   , growth=True),
            #     hist.axis.StrCategory([], name="systematic", growth=True), 
            #     hist.axis.Regular(50, 0, 100, name="met_uncertainty", label=r"$p_{T}^{miss} uncertainty$ (GeV)"),
            #     hist.storage.Weight()
            # ),
            'met_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="met_phi", label=r"$\phi^{miss}$"),
                hist.storage.Weight()
            ),
            'njets': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="njets", label=r"$N_{jet}$ ($p_{T}>30$ GeV)"),
                hist.storage.Weight()
            ), 
            'bjets': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="bjets", label=r"$N_{b-jet}$ ($p_{T}>30$ GeV)"),
                hist.storage.Weight()
            ),
            'dphi_met_ll': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="dphi_met_ll", label=r"$\Delta \phi(\ell\ell,p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            'gnn_score': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, 1, name="gnn_score", label=r"$O_{GNN}$"),
                hist.storage.Weight()
            ),
            'gnn_flat': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, 1, name="gnn_flat", label=r"$O_{GNN}$"),
                hist.storage.Weight()
            ),
            'dijet_mass': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, 2000, name="dijet_mass", label=r"$m_{jj}$ (GeV)"),
                hist.storage.Weight() 
            ),
            'dijet_deta': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(20, 0, 8, name="dijet_deta", label=r"$Delta\eta_{jj}$"),
                hist.storage.Weight()
            ),
            "lead_jet_pt": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="lead_jet_pt", label="$p_T^{j_1}$ (GeV)"),
                hist.storage.Weight()
            ), 
            "trail_jet_pt": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="trail_jet_pt", label=r"$p_T^{j_2}$ (GeV)"),
                hist.storage.Weight()
            ),
            "third_jet_pt": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="third_jet_pt", label=r"$p_T^{j_2}$ (GeV)"),
                hist.storage.Weight()
            ),
            "lead_jet_eta": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="lead_jet_eta", label=r"$\eta(j_1)$"),
                hist.storage.Weight()
            ), 
            "trail_jet_eta": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="trail_jet_eta", label=r"$\eta(j_2)$"),
                hist.storage.Weight()
            ),
            "third_jet_eta": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="third_jet_eta", label=r"$\eta(j_3)$"),
                hist.storage.Weight()
            ),
            "lead_jet_phi": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="lead_jet_phi", label=r"$\phi^(j_1)$"),
                hist.storage.Weight()
            ), 
            "trail_jet_phi": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="trail_jet_phi", label=r"$\phi^(j_2)$"),
                hist.storage.Weight()
            ),
            "third_jet_phi": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50,-np.pi, np.pi, name="third_jet_phi", label=r"$\phi^(j_3)$"),
                hist.storage.Weight()
            ),

            "leading_lep_pt": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="leading_lep_pt", label="$p_T^{l_1}$ (GeV)"),
                hist.storage.Weight()
            ), 
            "trailing_lep_pt": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="trailing_lep_pt", label=r"$p_T^{l_2}$ (GeV)"),
                hist.storage.Weight()
            ),
            "third_lep_pt": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="third_lep_pt", label=r"$p_T^{l_2}$ (GeV)"),
                hist.storage.Weight()
            ),
            "leading_lep_eta": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="leading_lep_eta", label=r"$\eta(l_1)$"),
                hist.storage.Weight()
            ), 
            "trailing_lep_eta": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="trailing_lep_eta", label=r"$\eta(l_2)$"),
                hist.storage.Weight()
            ),
            "third_lep_eta": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="third_lep_eta", label=r"$\eta(l_3)$"),
                hist.storage.Weight()
            ),
            "leading_lep_phi": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="leading_lep_phi", label=r"$\phi^(l_1)$"),
                hist.storage.Weight()
            ), 
            "trailing_lep_phi": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="trailing_lep_phi", label=r"$\phi^(l_2)$"),
                hist.storage.Weight()
            ),
            "third_lep_phi": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="third_lep_phi", label=r"$\phi^(l_3)$"),
                hist.storage.Weight()
            ),
            "min_dphi_met_j": hist.Hist( 
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="min_dphi_met_j", label=r"$\min\Delta\phi(p_{T}^{miss},j)$"),
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
        dataset = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        selection = PackedSelection()
        weights = Weights(len(event), storeIndividual=True)
        
        histos = self.build_histos()
        
        if is_data:
            selection.add('lumimask', self._json[self._era](event.run, event.luminosityBlock))
            selection.add('triggers', trigger_rules(event, self._triggers, self._era))

        
        # MET filters
        if "2016" in self._era:
            selection.add(
                'metfilter',
                #event.Flag.METFilters &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter &
                event.Flag.goodVertices &
                event.Flag.eeBadScFilter &
                event.Flag.BadPFMuonFilter &
                event.Flag.BadPFMuonDzFilter
            )
        else:
            selection.add(
                'metfilter',
                event.Flag.goodVertices &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter & 
                event.Flag.BadPFMuonFilter &
                event.Flag.BadPFMuonDzFilter &
                event.Flag.eeBadScFilter &
                event.Flag.ecalBadCalibFilter 
            )


        tight_lep, nloose_lep = build_leptons(
            event.Muon,
            event.Electron
        )
        
        had_taus = build_htaus(event.Tau, tight_lep)
        ntight_lep = ak.num(tight_lep)
        nhtaus_lep = ak.num(had_taus)
        jets = event.Jet

        overlap_leptons = ak.any(
            jets.metric_table(tight_lep) <= 0.4,
            axis=2
        )

        jet_mask = (
            ~overlap_leptons & 
            (jets.pt>30.0) & 
            (np.abs(jets.eta) < 4.7) &
            (jets.jetId >= 6) &   # tight JetID 7(2016) and 6(2017/8)
            ((jets.puId >= 6) | (jets.puId == 3) | (jets.pt >= 50)) # medium puID https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL 3,7 for 16and 16APV; 6,7 for 17,18
        )

        jet_mask_PUID = (
            ~overlap_leptons & 
            (jets.pt>30.0) & 
            (np.abs(jets.eta) < 4.7) & 
            (jets.jetId >= 6) # tight JetID 7(2016) and 6(2017/8)
        )
        
        jet_btag = (
            (event.Jet.btagDeepFlavB > btag_id(
                self.btag_wp,
                self._era + 'APV' if self._isAPV else self._era
            )) &
            (np.abs(jets.eta) < (2.4 if "2016" in self._era else 2.5))
        )
        good_jets = jets[~jet_btag & jet_mask]

        good_bjet = jets[jet_btag & jet_mask]
        good_jets_forBtag = jets[jet_mask & (np.abs(jets.eta) < (2.4 if "2016" in self._era else 2.5))]
        pu_good_jets = jets[~jet_btag & jet_mask_PUID]

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
        lead_lep = ak.firsts(dilep.l1,axis=1)
        subl_lep = ak.firsts(dilep.l2,axis=1)
        dilep_p4 = (lead_lep + subl_lep)
        dilep_m  = dilep_p4.mass
        dilep_pt = dilep_p4.pt
        third_lep = ak.firsts(extra_lep, axis=1)

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

        ptmiss_sigma = event.MET.significance
        
        # met_px = reco_met_pt * np.cos(reco_met_phi)
        # met_py = reco_met_pt * np.sin(reco_met_phi)

        # numerator = (
        #     met_px**2 * event.MET.covXX +
        #     2 * met_px * met_py * event.MET.covXY +
        #     met_py**2 * event.MET.covYY
        # )
        # ptmiss_unc = np.sqrt(numerator / (reco_met_pt**2))
	    
	    # this definition is not correct as it doesn't include the mass of the second Z
        dilep_et_ll = np.sqrt(dilep_pt**2 + dilep_m**2)
        dilep_et_met = np.sqrt(reco_met_pt**2 + self.zmass**2)
        
        # new version
        dilep_mt = ak.where(  
                ntight_lep==3,
                np.sqrt((dilep_et_ll + dilep_et_met)**2 - ((dilep_p4.pvec + emu_met.pvec).pt)**2),
                np.sqrt((dilep_et_ll + dilep_et_met)**2 - ((dilep_p4.pvec +  p4_met.pvec).pt)**2))
        

        dilep_dphi_met  = ak.where(ntight_lep==2, dilep_p4.delta_phi(p4_met), dilep_p4.delta_phi(emu_met))

        # 2jet and vbs related variables
        sorted_indices = np.argsort(-good_jets.pt)
        good_jets = good_jets[sorted_indices]
        
        lead_jet = ak.firsts(good_jets)
        subl_jet = ak.firsts(good_jets[lead_jet.delta_r(good_jets)>0.01])
        third_jet = ak.firsts(good_jets[(lead_jet.delta_r(good_jets)>0.01) & (subl_jet.delta_r(good_jets)>0.01)])

        leadbjet_score = lead_jet.btagDeepFlavB
        sublbjet_score = subl_jet.btagDeepFlavB
        event['leadbjet_score'] = ak.fill_none(leadbjet_score,np.nan)
        event['sublbjet_score'] = ak.fill_none(sublbjet_score,np.nan)
        
        dijet_mass = (lead_jet + subl_jet).mass
        dijet_deta = np.abs(lead_jet.eta - subl_jet.eta)
        event['dijet_mass'] = dijet_mass
        event['dijet_deta'] = dijet_deta 
        
        min_dphi_met_j = ak.min(np.abs(good_jets.delta_phi(p4_met)),axis=1)

        selection.add(
            "require-ossf",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            (ak.any(tight_lep.pt > 20)) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        
        selection.add(
            "require-osof",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            (ak.any(tight_lep.pt > 20)) &
            ((lead_lep.pdgId)*(subl_lep.pdgId) == -143)
        )
        
        selection.add(
            "require-3lep",
            (ntight_lep==3) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            (ak.any(tight_lep.pt > 20)) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        selection.add('met_pt' ,ak.fill_none(reco_met_pt > 120, False))
        selection.add('low_met_pt', ak.fill_none((reco_met_pt < 100) & (reco_met_pt > 50), False))
        selection.add('met_pt_120_200', ak.fill_none((reco_met_pt < 200) & (reco_met_pt > 120), False))
        selection.add('met_pt_200', ak.fill_none(reco_met_pt > 200, False))
        selection.add('medium_ptmiss', ak.fill_none((reco_met_pt > 70), False))
        selection.add('dilep_m'   , ak.fill_none(np.abs(dilep_m - self.zmass) < 15, False))
        selection.add('dilep_m_50', ak.fill_none(dilep_m > 50, False))
        selection.add(
            'dilep_pt',
            ak.where(
                selection.require(**{"require-3lep":True}),
                ak.fill_none(dilep_pt>45, False),
                ak.fill_none(dilep_pt>60, False)
            )
        )
        selection.add('dilep_pt_60_150',ak.fill_none((dilep_pt>60) & (dilep_pt<150), False))
        selection.add('dilep_pt_150_300',ak.fill_none((dilep_pt>150) & (dilep_pt<300), False))
        selection.add('dilep_pt_300_inf',ak.fill_none((dilep_pt>300), False))
        selection.add("dilep_dphi_met", ak.fill_none(np.abs(dilep_dphi_met)>1.0, False))
        selection.add("min_dphi_met_j",ak.fill_none(np.abs(min_dphi_met_j)>0.5, False))
        # jet demography
        # selection.add('1njets' , ngood_jets  >= 1 )
        selection.add('2njets' , ngood_jets  >= 2 )
        selection.add('1nbjets', ngood_bjets >= 1 )
        selection.add('0nhtaus', nhtaus_lep  == 0 )
        
        selection.add('dijet_deta', ak.fill_none(dijet_deta > 2.5, False))
        selection.add('dijet_mass_400' , ak.fill_none(dijet_mass >  400, False))
        selection.add('dijet_mass_400_low' , ak.fill_none(dijet_mass <  400, False))

        # Define all variables for the GNN
        event['met_sig'  ] = ak.fill_none(ptmiss_sigma,np.nan)
        # event['met_uncertainty'  ] = ak.fill_none(ptmiss_unc,np.nan)
        event['met_pt'  ] = ak.fill_none(reco_met_pt,np.nan)
        event['met_phi' ] = ak.fill_none(reco_met_phi,np.nan)
        event['dilep_mt'] = ak.fill_none(dilep_mt,np.nan)
        event['dilep_m'] = ak.fill_none(dilep_m,np.nan)
        event['dilep_pt'] = ak.fill_none(dilep_pt,np.nan)
        event['njets'   ] = ak.fill_none(ngood_jets,np.nan)
        event['bjets'   ] = ak.fill_none(ngood_bjets,np.nan)
        event['dphi_met_ll'] = ak.fill_none(dilep_dphi_met,np.nan)
        event['dijet_mass'] = ak.fill_none(dijet_mass,np.nan)
        event['dijet_deta'] = ak.fill_none(dijet_deta,np.nan)
        event['min_dphi_met_j'] = ak.fill_none(min_dphi_met_j,np.nan)

        event['leading_lep_pt'  ] = ak.fill_none(lead_lep.pt,np.nan)
        event['leading_lep_pdgId' ] = ak.fill_none(lead_lep.pdgId,np.nan)
        event['leading_lep_eta' ] = ak.fill_none(lead_lep.eta,np.nan)
        event['leading_lep_phi' ] = ak.fill_none(lead_lep.phi,np.nan)
        event['trailing_lep_pt' ] = ak.fill_none(subl_lep.pt,np.nan)
        event['trailing_lep_eta'] = ak.fill_none(subl_lep.eta,np.nan)
        event['trailing_lep_phi'] = ak.fill_none(subl_lep.phi,np.nan)
        event['third_lep_pt'  ] = ak.fill_none(third_lep.pt,np.nan)
        event['third_lep_eta' ] = ak.fill_none(third_lep.eta,np.nan)
        event['third_lep_phi' ] = ak.fill_none(third_lep.phi,np.nan)
        event['lead_jet_pt'  ] = ak.fill_none(lead_jet.pt,np.nan)
        event['lead_jet_eta' ] = ak.fill_none(lead_jet.eta,np.nan)
        event['lead_jet_phi' ] = ak.fill_none(lead_jet.phi,np.nan)
        event['trail_jet_pt' ] = ak.fill_none(subl_jet.pt,np.nan)
        event['trail_jet_eta'] = ak.fill_none(subl_jet.eta,np.nan)
        event['trail_jet_phi'] = ak.fill_none(subl_jet.phi,np.nan)
        event['third_jet_pt' ] = ak.fill_none(third_jet.pt,np.nan)
        event['third_jet_eta'] = ak.fill_none(third_jet.eta,np.nan)
        event['third_jet_phi'] = ak.fill_none(third_jet.phi,np.nan)
        
        # Apply GNN event['gnn_score'] = applyGNN(event,self._era).get_nnscore()
        # event['gnn_score'] = applyGNN(event).get_nnscore()
        # score_for_flat = np.minimum(
        #     np.maximum(event['gnn_score'], self._gnn_score_min),
        #     self._gnn_score_max,
        # )
        # raw_score = self.gnn_flat_fnc(score_for_flat)
        # raw_score = np.nan_to_num(
        #     raw_score,
        #     nan=self._gnn_flat_min,
        #     posinf=self._gnn_flat_max,
        #     neginf=self._gnn_flat_min,
        # )
        # event['gnn_flat'] = np.minimum(
        #     np.maximum(raw_score, self._gnn_flat_min),
        #     self._gnn_flat_max,
        # )


        # Now adding weights
        if not is_data:
            weights.add('genweight', event.genWeight)
            self._btag.append_btag_sf(good_jets_forBtag, weights)#good_jets_forBtag
            self._jpSF.append_jetPU_sf(pu_good_jets, weights)

            self._purw.append_pileup_weight(weights, event.Pileup.nTrueInt)
            self._tauID.append_tauID_sf(had_taus, weights)
            self._add_trigger_sf(weights, lead_lep, subl_lep)
    
            weights.add (
                    'LeptonSF', 
                    lead_lep.SF*subl_lep.SF, 
                    lead_lep.SF_up*subl_lep.SF_up, 
                    lead_lep.SF_down*subl_lep.SF_down
            )
            _ones = np.ones(len(weights.weight()))
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
                
            # if 'LHEReweightingWeight' in event.fields and 'aQGC' in dataset:
            #     for i in range(1057):
            #         weights.add(f"eft_{self._eftnames[i]}", _ones, event.LHEReweightingWeight[:, i])
            # print(weights.weight(),'\n')
            # 2017 Prefiring correction weight
            if 'L1PreFiringWeight' in event.fields:
                weights.add("prefiring_weight", event.L1PreFiringWeight.Nom, event.L1PreFiringWeight.Dn, event.L1PreFiringWeight.Up)
            

        # selections
        if is_data:
            common_sel = ['triggers', 'lumimask', 'metfilter']
        else:
            common_sel = ['metfilter']
        channels = {
           # vector boson scattering
            "vbs-SR": common_sel + [
            'require-ossf', "2njets",
            'dilep_m', 'dilep_pt', '0nhtaus',
            'dilep_dphi_met', 'min_dphi_met_j', 'met_pt', '~1nbjets', 
            "dijet_deta", "dijet_mass_400"
        ],
            "vbs-3L": common_sel + [
            'require-3lep', 'dilep_m', 'dilep_pt','0nhtaus',
            'dilep_dphi_met', #'min_dphi_met_j',
            'medium_ptmiss', '~1nbjets', "2njets"
            ],
            "vbs-EM": common_sel + [
            'require-osof',
            'dilep_m', 'dilep_pt', 
            'dilep_dphi_met', #'min_dphi_met_j',
            'medium_ptmiss', '~1nbjets','0nhtaus',
            "2njets"
            ],
        }

        if not is_data:
            vbs_sr_sel = channels["vbs-SR"]
            vbs_sr_args = {
                s.replace('~', ''): (False if '~' in s else True) for s in vbs_sr_sel
            }
            vbs_sr_mask = selection.require(**vbs_sr_args)
            dataDrivenDYRatio(
                dilep_pt,
                reco_met_pt,
                self._isDY,
                self._era,
                self._ddtype,
            ).ddr_add_weight(weights, target_mask=vbs_sr_mask)
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
                print(f" - {syst} weight nan/inf:", vv[np.isnan(vv)], vv[np.isinf(vv)])
            histos[var].fill(
                **{
                    "channel": ch, 
                    "systematic": systname, 
                    var: _format_variable(event[var], cut), 
                    "weight": ak.nan_to_num(weight,nan=1, posinf=1, neginf=1)
                        #ak.ones_like(weight)
                }
            )
            
        def _gnn_dumper(ch):
            sel_ = channels[ch]
            sel_args_ = {
                s.replace('~',''): (False if '~' in s else True) for s in sel_
            }
            cut =  selection.require(**sel_args_)
            weight = weights.weight()[cut]

            jet_pts = ak.to_list(ak.fill_none(good_jets[cut].pt, np.nan))

            _dicv = {
                ch: {
                    "event": _format_variable(event.event, cut).tolist(), 
                    # "gnn": _format_variable(event["gnn_score"], cut).tolist(), 
                    "jet_pt": jet_pts,
                    "lepton_pt": _format_variable(event['leading_lep_pt'], cut).tolist(), 
                    "lepton_pdgId" : _format_variable(event['leading_lep_pdgId'], cut).tolist(), 
                    "weight": weight.tolist()
                }
            }
            if 'gnn_dump' in histos:
                histos["gnn_dump"].update(_dicv)
            else:
                histos["gnn_dump"] = _dicv

        if shift_name is None:
            systematics = [None] + list(weights.variations)

        else:
            systematics = [shift_name]
        for ch in channels:
            if self.dump_gnn_array:
                _gnn_dumper(ch)
            for sys in systematics:
                # print(ch,sys)
                _histogram_filler(ch, sys, 'met_pt')
                _histogram_filler(ch, sys, 'dilep_mt')
                _histogram_filler(ch, sys, 'dilep_pt')
                _histogram_filler(ch, sys, 'dilep_m')
                _histogram_filler(ch, sys, 'njets')
                _histogram_filler(ch, sys, 'bjets')
                _histogram_filler(ch, sys, 'dphi_met_ll')
                _histogram_filler(ch, sys, 'dijet_mass')
                _histogram_filler(ch, sys, 'dijet_deta')
                _histogram_filler(ch, sys, 'min_dphi_met_j')
                # _histogram_filler(ch, sys, 'gnn_score')
                # _histogram_filler(ch, sys, 'gnn_flat')
                _histogram_filler(ch, sys,  'lead_jet_pt')
                _histogram_filler(ch, sys,  'trail_jet_pt')
                _histogram_filler(ch, sys,  'third_jet_pt')
                _histogram_filler(ch, sys,  'lead_jet_eta')
                _histogram_filler(ch, sys,  'trail_jet_eta')
                _histogram_filler(ch, sys,  'third_jet_eta')
                _histogram_filler(ch, sys,  'lead_jet_phi')
                _histogram_filler(ch, sys,  'trail_jet_phi')
                _histogram_filler(ch, sys,  'third_jet_phi')
                _histogram_filler(ch, sys,  'leading_lep_pt')
                _histogram_filler(ch, sys,  'trailing_lep_pt')
                _histogram_filler(ch, sys,  'third_lep_pt')
                _histogram_filler(ch, sys,  'leading_lep_eta')
                _histogram_filler(ch, sys,  'trailing_lep_eta')
                _histogram_filler(ch, sys,  'third_lep_eta')
                _histogram_filler(ch, sys,  'leading_lep_phi')
                _histogram_filler(ch, sys,  'trailing_lep_phi')
                _histogram_filler(ch, sys,  'third_lep_phi') 
        return {dataset: histos}
        
    def process(self, event: processor.LazyDataFrame):
        dataset_name = event.metadata['dataset']
        is_data = event.metadata.get("is_data")


        if is_data: 
            softjet_gen_pt = None
        else:
            softjet_gen_pt = find_best_match(event.CorrT1METJet,event.GenJet)
        
        rho = event.fixedGridRhoFastjetAll
        cache = event.caches[0]

        if not is_data:
            jets = self._jmeu.corrected_jets_jer(event.Jet, rho, event.caches[0])
            event = ak.with_field(event, jets, 'Jet')
            jets_shift_L123 = self._jmeu.corrected_jets_L123(event.Jet, rho, cache)
        else:
            jets = event.Jet
            emFraction = jets.chEmEF + jets.neEmEF
            mask_jec = (jets.pt > 15) & (emFraction <= 0.9)
            jets_shift_L123 = ak.mask(jets,mask_jec)
            jets_shift_L123["pt"] = jets_shift_L123.pt * (1 - jets.muonSubtrFactor)

        jets_shift_L1 = self._jmeu.corrected_jets_L1(event.Jet, rho, cache)

        softjets_shift_L123 = self._jmeu.corrected_jets_L123(event.CorrT1METJet, rho, cache, softjet_gen_pt)
        softjets_shift_L1 = self._jmeu.corrected_jets_L1(event.CorrT1METJet, rho, cache, softjet_gen_pt)
        jets_col_shift_L123 = ak.concatenate([jets_shift_L123, softjets_shift_L123],axis=1)
        jets_col_shift_L1 = ak.concatenate([jets_shift_L1, softjets_shift_L1],axis=1)
            
        
        raw_met = event.RawMET
        met_to_correct = event.MET
        met_to_correct["pt"] = raw_met.pt
        met_to_correct["phi"] = raw_met.phi
        met = self._jmeu.corrected_met(met_to_correct, jets_col_shift_L123, jets_col_shift_L1, rho, event.caches[0])

        # x-y met shit corrections

        run = event.run 
        npv = event.PV.npvs
        
        met = met_phi_xy_correction(
            met, run, npv, 
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

        electronEnUp  ['pt'] = event.Electron['pt'] + event.Electron.energyErr/np.cosh(event.Electron.eta)
        electronEnDown['pt'] = event.Electron['pt'] - event.Electron.energyErr/np.cosh(event.Electron.eta)	
	
        hem_overlap = None
        if (self._era == '2018') and (not is_data):
            tight_lep_for_hem, _ = build_leptons(event.Muon, event.Electron)
            hem_overlap = ak.any(
                event.Jet.metric_table(tight_lep_for_hem) <= 0.4,
                axis=2
            )

        # define all the shifts
        shifts = [
            # Jets
            ({"Jet": event.Jet                             , "MET": event.MET                               }, None                  ),
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
            # year dependent systematics
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
        ]

        if (self._era == '2018') and (not is_data):
            hem_jets, hem_met = apply_hem_uncertainty(
                event.Jet,
                event.MET,
                overlap_leptons=hem_overlap
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
# samples ={
#        "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8":{
#            'files': [
#                "/tmp/hgao/ZZTo2L2Nu_p0_Dilepton-MC_17.root",
# ],
#            'metadata':{
#                'era': "2018",
#                'is_data': False
#            }
#        }
#    }
# out_btag1 = processor.run_uproot_job(
#     samples,
#     processor_instance=zzinc_processor(
#         era='2018',
#         isDY=True,
#         ewk_process_name="ZZ",
#         run_period='',
#         dump_gnn_array=False),
#     treename='Events',
#     executor=processor.futures_executor,
#     executor_args={
#         "schema": nanoevents.NanoAODSchema,
#         "workers": 16
#     },
#     # chunksize=200,
#     # maxchunks=2
# )

# print(out_btag1['ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8']['met_pt'][{'channel': 'vbs-SR', 'systematic': 'nominal'}].values().sum())

