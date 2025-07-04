import gzip 
import pickle
import argparse
import collections
import functools
import itertools
import os, glob
import subprocess
from tqdm import tqdm
import numpy as np
def updated(c, items):
    c.update(items)
    return c
tag = "vbs"
eras = ["2016","2016APV","2018","2017"]
eras = ["2018"]

for era in eras:
    all_hists = []
    #for filename in tqdm(glob.glob(f'/eos/user/h/hgao/ZZTo2L2Nu/PKL/{era}_noDD_1129/*.pkl.gz'), desc="reading", ascii=False, ncols=75):
    for filename in tqdm(glob.glob(f'/afs/cern.ch/user/h/hgao/SMQawa/{era}-SR-v2/*/*.pkl.gz'), desc="reading", ascii=False, ncols=75):
        if os.path.getsize(filename) == 0: 
            print(f"{filename} is empty !! ")
        else:
            with gzip.open(filename, 'rb') as f:
                _data = pickle.load(f)
                all_hists.append(_data)
                f.close()
                del f 
                del _data
            
    combined_hist = {}
    combined_sumw = {}
    
    for i in tqdm(all_hists, desc="format", ascii=False, ncols=75):
        for s, v in i.items():
            v_hist = dict(filter(lambda n: not isinstance(n[1], dict), v['hist'].items()))
            if s in combined_hist:
                combined_hist[s].append(v_hist)
                combined_sumw[s].append(v['sumw'])
            else:
                combined_hist[s] = [v_hist]
                combined_sumw[s] = [v["sumw"]]
    
    del all_hists

    combined_dict = {}
    for s, h in tqdm(combined_hist.items(), desc="merging", ascii=False, ncols=75):
        hist_ = dict(functools.reduce(updated, h, collections.Counter()))
        sumw_ = sum(combined_sumw[s])
        combined_dict[s] = {"hist": hist_, "sumw": sumw_}
    with gzip.open(f"/eos/user/h/hgao/ZZTo2L2Nu/PKL/{era}_0519_DY_addDD.pkl.gz", "wb") as f:
        pickle.dump(combined_dict, f)
        f.close()
        
 
