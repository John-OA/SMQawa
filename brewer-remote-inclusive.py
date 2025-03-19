from coffea import processor
from coffea import nanoevents
from coffea.nanoevents import NanoAODSchema, BaseSchema
from qawa.process.wztau2lnu_inclusive import wzinclusive_processor
from qawa.process.coffea_sumw import coffea_sumw
import argparse
import pickle
import gzip
import re, sys
import uproot
import numpy as np

NanoAODSchema.warn_missing_crossrefs = False

np.seterr(all='ignore')

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
uproot.open.defaults["timeout"] = 60*10 # wait more


def validate_input_file(nanofile):
    pfn = nanofile
    pfn=re.sub("\n","",pfn)
    aliases = [
        "root://eoscms.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cmsxrootd.fnal.gov/",
        "root://cms-xrd-global.cern.ch/",
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
            print(f'--> {alias} FAILED')

        if valid==False:
            # all faild force AAA anyways
            nanofile = aliases[-1] + pfn
    return nanofile

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--jobNum' ,   type=int, default=1     , help="")
    parser.add_argument('--era'    ,   type=str, default="2018", help="")
    parser.add_argument('--isMC'   ,   type=int, default=1     , help="")
    parser.add_argument('--infile' ,   type=str, default=None  , help="input root file")
    parser.add_argument('--dataset',   type=str, default=None  , help="dataset name. need to specify if file is not in EOS")
    parser.add_argument('--runperiod', type=str, default=None)
    parser.add_argument('--executor' , type=str, default="FuturesExecutor", help="Executor to use, one of IterativeExecutor (good for debugging), FuturesExecutor (multithreaded), or other coffea option")

    options = parser.parse_args()

    if options.dataset is None: 
        options.dataset = options.infile.split('/')[4]
    executor = None
    if options.executor == "FuturesExecutor":
        executor = processor.FuturesExecutor(workers=8,)
    elif options.executor == "IterativeExecutor":
        executor = processor.IterativeExecutor()
    elif options.executor == "DaskExecutor":
        executor = processor.DaskExecutor()
    else:
        raise ValueError(f"Invalid Executor option {options.executor}")

    era=options.era
    is_data = not options.isMC

    failed = True
    ixrd = 0
    aliases = [
        "root://eoscms.cern.ch/",
        "root://llrxrd-redir.in2p3.fr/",
        "root://xrootd-cms.infn.it/",
        "root://cms-xrd-global01.cern.ch/", 
        "root://cms-xrd-global02.cern.ch/",
        "root://cmsxrootd.fnal.gov/",
        "root://xrootd-cms-redir-int.cr.cnaf.infn.it/",
        "root://xrootd-redic.pi.infn.it/"
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

            sumw_runner = processor.Runner(
                executor=executor,
                schema=BaseSchema,
                format="root",
            )
            sumw_out = sumw_runner(samples,
                                   "Runs",
                                   processor_instance=coffea_sumw(),
                                   )
            
            ewk_flag = None
            if "ZZTo" in options.infile and "GluGluTo" not in options.infile and "ZZJJ" not in options.infile:
                ewk_flag= 'ZZ'
            if "WZTo" in options.infile and "GluGluTo" not in options.infile:
                ewk_flag = 'WZ'

            # extarct the run period
            if is_data:
                if 'Run20' in options.infile:
                    options.runperiod = file_name.split('/store/data/')[1].split('/')[0].replace(f'Run{options.era}','')
            else:
                options.runperiod = ''

            print(
                f"""---------------------------
                -- options  = {options}
                -- is MC    = {options.isMC}
                -- jobNum   = {options.jobNum}
                -- era      = {options.era}
                -- in file  = {aliases[ixrd] + options.infile}
                -- dataset  = {options.dataset}
                -- period   = {options.runperiod}
                -- executor = {options.executor}
                ---------------------------"""
            )

            print(" --- wztau2lnu_inclusive processor ... ")
            vbs_runner = processor.Runner(
                executor=executor,
                schema=NanoAODSchema,
                chunksize=100000,
                # maxchunks=5
                format="root",
            )
            vbs_out = vbs_runner(samples,
                                 "Events",
                                 processor_instance=wzinclusive_processor(
                                     era=options.era,
                                     ewk_process_name=ewk_flag,
                                     run_period=options.runperiod if is_data else ''
                                 ),
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
        except Exception as e:
            print(e)
            print(f"[WARNING] {aliases[ixrd]} failed with the following error : ")
            print(sys.exc_info()[0])
            #print(f"Unexpected {err=}, {type(err)=}")
            print("-------------------------------------------")
            failed=True
            ixrd += 1
            if ixrd > (len(aliases) - 1):
                break

if __name__ == "__main__":
    main()
