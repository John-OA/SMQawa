import os.path
from coffea.lookup_tools import extractor, dense_lookup
import awkward as ak
import numpy as np
import uproot 


class LeptonScaleFactors:
    def __init__(self, era:str='2018', isAPV:bool=False):
        if isAPV:
            self._era = era + 'APV'
        else:
            self._era = era 
        extLepSF = extractor()

        _data_path = os.path.join(os.path.dirname(__file__), 'data/LeptonSF')

        muonSelectionTag     = "TightWP_" + self._era
        electronSelectionTag = "GPMVA90_" + self._era

        if muonSelectionTag=="TightWP_2016":
            mu_f=["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.root"]
            mu_h = ["NUM_TightID_DEN_TrackerMuons_abseta_pt",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
            mu_f_up = ["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID_variations.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO_variations.root"]
            mu_h_up = ['NUM_LooseID_DEN_TrackerMuons_abseta_pt_up;1',
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_up;1"]
            mu_f_down = ["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID_variations.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO_variations.root"]
            mu_h_down = ["NUM_LooseID_DEN_TrackerMuons_abseta_pt_down;1",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_down"]

        elif muonSelectionTag=="TightWP_2016APV":
            mu_f=["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.root"]
            mu_h = ["NUM_TightID_DEN_TrackerMuons_abseta_pt",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
            mu_f_up = ["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID_variations.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO_variations.root"]
            mu_h_up = ['NUM_LooseID_DEN_TrackerMuons_abseta_pt_up;1',
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_up;1"]
            mu_f_down = ["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID_variations.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO_variations.root"]
            mu_h_down = ["NUM_LooseID_DEN_TrackerMuons_abseta_pt_down;1",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_down"]
        elif muonSelectionTag=="TightWP_2017":
            mu_f=["2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.root",
                  "2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.root"]
            mu_h = ["NUM_TightID_DEN_TrackerMuons_abseta_pt",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
            mu_f_up = ["2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID_variations.root",
                  "2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO_variations.root"]
            mu_h_up = ['NUM_LooseID_DEN_TrackerMuons_abseta_pt_up;1',
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_up;1"]
            mu_f_down = ["2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID_variations.root",
                  "2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO_variations.root"]
            mu_h_down = ["NUM_LooseID_DEN_TrackerMuons_abseta_pt_down;1",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_down"]
        elif muonSelectionTag=="TightWP_2018":
            mu_f=["2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.root",
                  "2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.root"]
            mu_h = ["NUM_TightID_DEN_TrackerMuons_abseta_pt",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
            mu_f_up = ["2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID_variations.root",
                  "2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO_variations.root"]
            mu_h_up = ['NUM_LooseID_DEN_TrackerMuons_abseta_pt_up;1',
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_up;1"]
            mu_f_down = ["2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID_variations.root",
                  "2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO_variations.root"]
            mu_h_down = ["NUM_LooseID_DEN_TrackerMuons_abseta_pt_down;1",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_down"]
        else:
            print (f'wrong era: {muonSelectionTag}')


        if electronSelectionTag=="GPMVA90_2016":
            el_f = ["2016/egammaEffi.txt_Ele_wp90iso_postVFP_EGM2D.root",
                    "2016/electron_RecoSF_UL2016postVFP.root"]
            el_h = ["EGamma_SF2D","EGamma_SF2D"]
            el_f_up = ["2016/egammaEffi.txt_Ele_wp90iso_postVFP_EGM2D_variations.root",
                    "2016/electron_RecoSF_UL2016postVFP.root"]
            el_h_up = ["EGamma_SF2D_up","EGamma_SF2D"]
            el_f_down = ["2016/egammaEffi.txt_Ele_wp90iso_postVFP_EGM2D_variations.root",
                    "2016/electron_RecoSF_UL2016postVFP.root"]
            el_h_down = ["EGamma_SF2D_down","EGamma_SF2D"]
        elif electronSelectionTag=="GPMVA90_2016APV":
            el_f = ["2016/egammaEffi.txt_Ele_wp90iso_preVFP_EGM2D.root",
                    "2016/electron_RecoSF_UL2016preVFP.root"]
            el_h = ["EGamma_SF2D","EGamma_SF2D"]
            el_f_up = ["2016/egammaEffi.txt_Ele_wp90iso_preVFP_EGM2D_variations.root",
                    "2016/electron_RecoSF_UL2016preVFP.root"]
            el_h_up = ["EGamma_SF2D_up","EGamma_SF2D"]
            el_f_down = ["2016/egammaEffi.txt_Ele_wp90iso_preVFP_EGM2D_variations.root",
                    "2016/electron_RecoSF_UL2016preVFP.root"]
            el_h_down = ["EGamma_SF2D_down","EGamma_SF2D"]
        elif electronSelectionTag=="GPMVA90_2017":
            el_f = ["2017/egammaEffi.txt_EGM2D_MVA90iso_UL17.root",
                    "2017/electron_RecoSF_UL2017.root"]
            el_h = ["EGamma_SF2D","EGamma_SF2D"]
            el_f_up = ["2017/egammaEffi.txt_EGM2D_MVA90iso_UL17_variations.root",
                    "2017/electron_RecoSF_UL2017.root"]
            el_h_up = ["EGamma_SF2D_up","EGamma_SF2D"]
            el_f_down = ["2017/egammaEffi.txt_EGM2D_MVA90iso_UL17_variations.root",
                    "2017/electron_RecoSF_UL2017.root"]
            el_h_down = ["EGamma_SF2D_down","EGamma_SF2D"]
        elif electronSelectionTag=="GPMVA90_2018":
            el_f = ["2018/egammaEffi.txt_Ele_wp90iso_EGM2D.root",
                    "2018/electron_RecoSF_UL2018.root"]
            el_h = ["EGamma_SF2D","EGamma_SF2D"]
            el_f_up = ["2018/egammaEffi.txt_Ele_wp90iso_EGM2D_variations.root",
                    "2018/electron_RecoSF_UL2018.root"]
            el_h_up = ["EGamma_SF2D_up","EGamma_SF2D"]
            el_f_down = ["2018/egammaEffi.txt_Ele_wp90iso_EGM2D_variations.root",
                    "2018/electron_RecoSF_UL2018.root"]
            el_h_down = ["EGamma_SF2D_down","EGamma_SF2D"]
        else:
            print (f'wrong era: {electronSelectionTag}')

        self.maps_nom = {}
        self.maps_up = {}
        self.maps_down = {}
        for i, _fname in enumerate(mu_f):
            with uproot.open(f"{_data_path}/{_fname}") as _fn:
                _hist = _fn[mu_h[i]].to_hist()
                _hnom = _hist.values()
                tag = f"MuonIso{era}" if "ISO" in _fname else ""
                tag = f"MuonId{era}"  if "ID" in _fname else tag
                self.maps_nom[tag] = dense_lookup.dense_lookup(_hnom,[ax.edges for ax in _hist.axes])
            with uproot.open(f"{_data_path}/{mu_f_up[i]}") as _fn:
                _hist = _fn[mu_h_up[i]].to_hist()
                _herr_up = _hist.values()
                tag = f"MuonIso{era}" if "ISO" in _fname else ""
                tag = f"MuonId{era}"  if "ID" in _fname else tag
                self.maps_up[tag] = dense_lookup.dense_lookup(_herr_up,[ax.edges for ax in _hist.axes])
            with uproot.open(f"{_data_path}/{mu_f_down[i]}") as _fn:
                _hist = _fn[mu_h_down[i]].to_hist()
                _herr_down = _hist.values()
                tag = f"MuonIso{era}" if "ISO" in _fname else ""
                tag = f"MuonId{era}"  if "ID" in _fname else tag
                self.maps_down[tag] = dense_lookup.dense_lookup(_herr_down,[ax.edges for ax in _hist.axes])

        for i, _fname in enumerate(el_f):
            with uproot.open(f"{_data_path}/{_fname}") as _fn:
                _hist = _fn[el_h[i]].to_hist()
                _hnom = _hist.values()
                tag = f"ElectronIso{era}" if "iso" in _fname else ""
                tag = f"ElectronReco{era}" if "Reco" in _fname else tag
                self.maps_nom[tag] = dense_lookup.dense_lookup(_hnom,[ax.edges for ax in _hist.axes])
            with uproot.open(f"{_data_path}/{el_f_up[i]}") as _fn:
                _hist = _fn[el_h_up[i]].to_hist()
                _herr_up = _hist.values()
                tag = f"ElectronIso{era}" if "iso" in _fname else ""
                tag = f"ElectronReco{era}" if "Reco" in _fname else tag
                self.maps_up[tag] = dense_lookup.dense_lookup(_herr_up,[ax.edges for ax in _hist.axes])
            with uproot.open(f"{_data_path}/{el_f_down[i]}") as _fn:
                _hist = _fn[el_h_down[i]].to_hist()
                _herr_down = _hist.values()
                tag = f"ElectronIso{era}" if "iso" in _fname else ""
                tag = f"ElectronReco{era}" if "Reco" in _fname else tag
                self.maps_down[tag] = dense_lookup.dense_lookup(_herr_down,[ax.edges for ax in _hist.axes])


    def muonSF(self, muons: ak.Array):
        sf_nom  = 1.0 # ak.ones_like(muons.pt)
        sf_up   = 1.0 # ak.ones_like(muons.pt)
        sf_down = 1.0 # ak.ones_like(muons.pt)

        for n in self.maps_nom.keys():
            if 'Muon' not in n: continue
            _nom = self.maps_nom[n](muons.pt, np.abs(muons.eta))
            _up = self.maps_up[n](muons.pt, np.abs(muons.eta))
            _down = self.maps_down[n](muons.pt, np.abs(muons.eta))
            sf_nom = sf_nom * _nom 
            sf_up = sf_up * _up
            sf_down = sf_down * _down
            
        return sf_nom, sf_up, sf_down

    def electronSF(self, electrons: ak.Array):
        sf_nom  = 1.0 # ak.ones_like(muons.pt)
        sf_up   = 1.0 # ak.ones_like(muons.pt)
        sf_down = 1.0 # ak.ones_like(muons.pt)
        
        for n in self.maps_nom.keys():
            if 'Electron' not in n: continue
            _nom = self.maps_nom[n](electrons.pt, np.abs(electrons.eta))
            _up = self.maps_up[n](electrons.pt, np.abs(electrons.eta))
            _down = self.maps_down[n](electrons.pt, np.abs(electrons.eta))
            sf_nom = sf_nom * _nom 
            sf_up = sf_up * _up
            sf_down = sf_down * _down
            
        return sf_nom, sf_up, sf_down
