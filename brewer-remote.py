from coffea import processor
from coffea import nanoevents
from qawa.process.zz2l2nu_vbs import zzinc_processor
from qawa.process.coffea_sumw import coffea_sumw
import argparse
import pickle
import gzip
import re, sys
import uproot
import numpy as np

np.seterr(all='ignore')

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
uproot.open.defaults["timeout"] = 60 * 5 # wait more


def validate_input_file(nanofile):
    pfn = nanofile
    pfn=re.sub("\n","",pfn)
    aliases = [
        "",
        # "root://eoscms.cern.ch/",
        # "root://xrootd-cms.infn.it/",
        # "root://cmsxrootd.fnal.gov/"
        # "root://cms-xrd-global.cern.ch/",
    ]

    valid = False
    for alias in aliases:
        testfile = None
        try:
            testfile=uproot.open(alias + pfn)
        except:
            pass
        if testfile:
            nanofile=alias + pfn
            print(f'--> {alias} OK')
            valid = True
            break
        else:
            print(f'--> {alias} FAILD')

        if valid==False:
            # all faild force AAA anyways
            nanofile = aliases[-1] + pfn
    return nanofile

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--jobNum' ,   type=int, default=1     , help="")
    parser.add_argument('--era'    ,   type=str, default="2018", help="")
    parser.add_argument('--isMC'   ,   type=int, default=1     , help="")
    parser.add_argument("-d"   , "--dd"    , type=str, default="onlySR"     , help="onlySR,DYSR,MC", required=True)
    parser.add_argument('--infile' ,   type=str, default=None  , help="input root file")
    parser.add_argument('--dataset',   type=str, default=None  , help="dataset name. need to specify if file is not in EOS")
    parser.add_argument('--runperiod', type=str, default=None)
    parser.add_argument('--dumpgnn', action='store_true', help='dump the GNN output into outputfiles')

    options = parser.parse_args()

    if options.dataset is None:
        if options.infile.split('/')[1] == 'eos':
            if options.infile.split('/')[2] == 'user':
                if options.isMC:
                    options.dataset = options.infile.split('/')[7] #for none aQGC it's [4], for aQGC and skimmed tree it's[7]
                else:
                    options.dataset = options.infile.split('/')[7].split('_')[0]
            elif options.infile.split('/')[2] == 'cms':
                if options.isMC:
                    options.dataset = options.infile.split('/')[9] #for none aQGC it's [4], for aQGC and skimmed tree it's[7]
                else:
                    options.dataset = options.infile.split('/')[9].split('_')[0]
        else:
            options.dataset = options.infile.split('/')[4]

    era=options.era
    is_data = not options.isMC

    # extarct the run period
    # skimed tree input example: /eos/user/y/yixiao/HZZsample/2018/DoubleMuon_Run2018A-UL2018_MiniAODv2_NanoAODv9-v1/DoubleMuon_p0_Dilepton-DoubleMuon_1.root
    if is_data and options.runperiod is None:
        if 'Run20' in options.infile:
            if '/store/data/' in options.infile:
                options.runperiod = options.infile.split('/store/data/')[1].split('/')[0].replace(f'Run{options.era}','')
            elif '/eos/' in options.infile:
                options.runperiod = options.infile.split(f'/{era}/')[1].split('-')[0].split('_')[1].replace(f'Run{options.era}','')
        else:
            options.runperiod = ''

    failed = True
    ixrd = 0
    aliases = [
        "",
        "root://eoscms.cern.ch/",
        # "root://llrxrd-redir.in2p3.fr/",
        # "root://xrootd-cms.infn.it/",
        # "root://cms-xrd-global01.cern.ch/", 
        # "root://cms-xrd-global02.cern.ch/",
        # "root://cmsxrootd.fnal.gov/",
        # "root://xrootd-cms-redir-int.cr.cnaf.infn.it/",
        # "root://xrootd-redic.pi.infn.it/"
    ]
    while failed:
        try:
            file_name = options.infile
            if '/store/' in options.infile:
                file_name = aliases[ixrd] + options.infile
            else:
                file_name = options.infile 

            samples ={
                options.dataset:{
                    'files': [file_name],
                    'metadata':{
                        'era': era,
                        'is_data': is_data
                    }
                }
            }
            
            
            sumw_out = processor.run_uproot_job(
                samples,
                treename="Runs",
                processor_instance=coffea_sumw(),
                executor=processor.futures_executor,
                executor_args={
                    "schema" : nanoevents.BaseSchema,
                    "workers": 8,
                    "desc": "SumW"
                },
                # chunksize=10000,
                # maxchunks=30
            )
            
            ewk_flag = None
            if "ZZTo2L2Nu" in options.dataset and "GluGluTo" not in options.dataset and "ZZJJ" not in options.dataset:
                ewk_flag= 'ZZ'
            if "WZTo" in options.dataset and "GluGluTo" not in options.dataset:
                ewk_flag = 'WZ'

            DY_flag = False
            if options.isMC:
                if "DYJetsToLL" in options.infile:
                    DY_flag = True

            print(
                f"""---------------------------
                -- options  = {options}
                -- is MC    = {options.isMC}
                -- jobNum   = {options.jobNum}
                -- era      = {options.era}
                -- in file  = {aliases[ixrd] + options.infile}
                -- dataset  = {options.dataset}
                -- period   = {options.runperiod}
                -- ewk_flag = {ewk_flag}
                -- isDY     = {DY_flag}
                ---------------------------"""
            )

            print(" --- zz2l2nu_vbs processor ... ",samples)
            vbs_out = processor.run_uproot_job(
                samples,
                processor_instance=zzinc_processor(
                    era=options.era,
                    isDY=DY_flag,
                    dd = options.dd,
                    ewk_process_name=ewk_flag,
                    dump_gnn_array=options.dumpgnn, 
                    run_period=options.runperiod if is_data else ''
                ),
                treename='Events',
                executor=processor.futures_executor,
                executor_args={
                    "schema" : nanoevents.NanoAODSchema,
                    "workers": 8,
                    "desc": "ZZinC"
                },
                # chunksize=10,
                # maxchunks=2
            )
            bh_output = {}
            for key, content in vbs_out.items():
                bh_output[key] = {
                    "hist": content,
                    "sumw": sumw_out[key],
            }
            with gzip.open("histogram_%s.pkl.gz" % str(options.jobNum), "wb") as f:
                pickle.dump(bh_output, f)
            failed=False
        except Exception as err:
            print(f"[WARNING] {aliases[ixrd]} failed with the following error : ")
            print(f"Unexpected {err=}, {type(err)=}")
            print("-------------------------------------------")
            failed=True
            ixrd += 1
            if ixrd > (len(aliases) - 1):
                break

if __name__ == "__main__":
    main()
