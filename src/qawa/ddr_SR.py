import pickle
import awkward as ak
import numpy as np
import os

from coffea.analysis_tools import Weights

class dataDrivenDYRatio:
    def __init__(self, dilep_pt, met_pt, isDY=False, era:str='2018', ddtype:str='onlySR'):
        self._isDY =isDY
        self._era = era
        self.met_pt = met_pt
        self.dilep_pt = dilep_pt

        _data_path = os.path.join(os.path.dirname(__file__), 'data')
        with open(f"{_data_path}/ddr/DDMC_Ratio_SR_{era}.pkl", 'rb') as pkl_file:
            ddmc_ratio = pickle.load(pkl_file)

        self.ddmc_ratio = ddmc_ratio

    def ddr_add_weight(self, weights:Weights, target_mask=None):

        ddmc_ratio = self.ddmc_ratio
        met_pt=self.met_pt
        dilep_pt=self.dilep_pt

        met_pt_below200 = ak.fill_none(met_pt < 200, False)
        met_pt_above200 = ak.fill_none(met_pt >=  200, False)

        if not self._isDY :
            dd_weights = np.zeros(len(met_pt))
            dd_weights_up = np.zeros(len(met_pt))
            dd_weights_down = np.zeros(len(met_pt))
        else:
            dd_weights = np.zeros(len(met_pt))
            dd_weights_up = np.zeros(len(met_pt))
            dd_weights_down = np.zeros(len(met_pt))
            for i in range(len(ddmc_ratio['dilep_pt_bin'])-1):
                for j in range(len(ddmc_ratio['met_pt_bin'])-1):
                    weights_i= np.array(list(map(lambda x: ddmc_ratio['ratio_mapping'][i][j] if x else 0,
                                                 (met_pt >= ddmc_ratio['met_pt_bin'][j])
                                                 & (met_pt < ddmc_ratio['met_pt_bin'][j+1])
                                                 & (dilep_pt >= ddmc_ratio['dilep_pt_bin'][i])
                                                 & (dilep_pt < ddmc_ratio['dilep_pt_bin'][i+1]))))
                    dd_weights = dd_weights + weights_i

        if '2016' in self._era:
            dd_weights_up[met_pt_below200]  = dd_weights[met_pt_below200] * 1.086
            dd_weights_down[met_pt_below200] = dd_weights[met_pt_below200] * 0.924
            dd_weights_up[met_pt_above200]   = dd_weights[met_pt_above200] * 1.277
            dd_weights_down[met_pt_above200] = dd_weights[met_pt_above200] * 0.757
        elif self._era in ['2018', '2017']:
            dd_weights_up[met_pt_below200]   = dd_weights[met_pt_below200] * 1.019
            dd_weights_down[met_pt_below200] = dd_weights[met_pt_below200] * 0.983
            dd_weights_up[met_pt_above200]  = dd_weights[met_pt_above200] * 1.281
            dd_weights_down[met_pt_above200] = dd_weights[met_pt_above200] * 0.754
        else:
            raise NotImplementedError(f"{self._era} not handled in dataDrivenDYRatio class")
        dd_weights[dd_weights==0] = 1
        dd_weights_up[dd_weights_up==0]=1
        dd_weights_down[dd_weights_down==0]=1

        if target_mask is not None:
            target_mask = np.asarray(target_mask, dtype=bool)
            dd_weights = np.where(target_mask, dd_weights, 1.0)
            dd_weights_up = np.where(target_mask, dd_weights_up, 1.0)
            dd_weights_down = np.where(target_mask, dd_weights_down, 1.0)

        weights.add(f'dataDrivenDYRatio_{self._era}', dd_weights, dd_weights_up, dd_weights_down)
