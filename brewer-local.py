from coffea import processor
from coffea import nanoevents
from qawa.process.zz2l2nu_vbs_agreeWithHZZ import zzinc_processor
from qawa.process.coffea_sumw import coffea_sumw
import argparse
import pickle
import gzip
import os, re
import uproot
import numpy as np
import json
np.seterr(all='ignore') #忽略浮点数运算中的错误
coffea_location = os.path.dirname(processor.__file__)
print("Coffea package is located at:", coffea_location)

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource #将默认的 XRootD 处理程序设置为多线程的 XRootD 处理程序
uproot.open.defaults["timeout"] = 60 * 30 # time out时间设置为10min


def validate_input_file(nanofile): #用于验证输入文件路径
    pfn = nanofile
    pfn=re.sub("\n","",pfn) #去除路径中可能存在的换行符
    aliases = [
        "",
        "root://eoscms.cern.ch/",
        "root://cms-xrd-global.cern.ch/",
        # "root://llrxrd-redir.in2p3.fr/",
        # "root://xrootd-cms.infn.it/",
        # "root://cms-xrd-global01.cern.ch/", 
        # "root://cms-xrd-global02.cern.ch/",
        # "root://cmsxrootd.fnal.gov/",
        # "root://xrootd-cms-redir-int.cr.cnaf.infn.it/",
        # "root://xrootd-redic.pi.infn.it/"
    ] #常见的远程数据存储位置的前缀

    valid = False
    for alias in aliases: #根据给定的别名列表尝试打开文件以验证文件路径的有效性，并选择有效的路径进行后续操作
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
    parser.add_argument('--jobNum', type=int, default=1, help="")
    parser.add_argument('-era'  , '--era' ,   type=str, default="2018", help="")
    parser.add_argument('-isMC' , '--isMC',   type=int, default=1, help="")
    parser.add_argument('-infile','--infile', type=str, default=None  , help="")
    
    options = parser.parse_args()
    dataset = options.infile.split('/')[4]
    #dataset = options.infile.split('/')[3]
    options.infile = validate_input_file(options.infile)
    era=options.era
    is_data = not options.isMC

    ixrd = 0
    aliases = [
        "",
	    "root://eoscms.cern.ch/",
        "root://cms-xrd-global.cern.ch/",
        # "root://llrxrd-redir.in2p3.fr/",
        # "root://xrootd-cms.infn.it/",
        # "root://cms-xrd-global01.cern.ch/",
        # "root://cms-xrd-global02.cern.ch/",
        # "root://cmsxrootd.fnal.gov/",
        # "root://xrootd-cms-redir-int.cr.cnaf.infn.it/",
        # "root://xrootd-redic.pi.infn.it/",
        # "root://cmsxrootd.fnal.gov/",
        # "root://cms-xrd-global.cern.ch/",
    ]
    #'files': [aliases[ixrd] + options.infile],
    samples ={
        dataset:{
            'files': [options.infile],
            'metadata':{
                'era': era,
                'is_data': is_data
            }
        }
    }
    

    print(
        "---------------------------"
        f"-- options  = {options}"
        f"-- is MC    = {options.isMC}"
        f"-- jobNum   = {options.jobNum}"
        f"-- era      = {options.era}"
        f"-- in file  = {aliases[ixrd] + options.infile}"
        f"-- dataset  = {dataset}"
        "---------------------------"
    )

    sumw_out = processor.run_uproot_job(
        samples,
        treename="Runs",
        processor_instance=coffea_sumw(),
        executor=processor.futures_executor,
        executor_args={
            "schema" : nanoevents.BaseSchema,
            "workers": 16,
        },
        chunksize=160,
        maxchunks=2
    )#返回计算得到的样本权重总和，并将其赋值给 sumw_out 变量
    
    ewk_flag = None
    if "ZZTo" in options.infile and "GluGluTo" not in options.infile and "ZZJJ" not in options.infile:
        ewk_flag= 'ZZ'
    if "WZTo" in options.infile and "GluGluTo" not in options.infile:
        ewk_flag = 'WZ'
        
    # extarct the run period
    run_period = '' 
    if 'Run20' in options.infile and is_data:
        run_period = options.infile.split('/store/data/')[1].split('/')[0].replace(f'Run{options.era}','')
        #run_period = options.infile
    print(" --------------------------- ")
    vbs_out = processor.run_uproot_job(
        samples,
        processor_instance=zzinc_processor(
            era=options.era,
            ewk_process_name=ewk_flag,
            dump_gnn_array=True,
            run_period=run_period if is_data else ''
        ),
        treename='Events',
        executor=processor.futures_executor,
        executor_args={
            "schema" : nanoevents.NanoAODSchema,
            "workers": 16,
        },
        chunksize=16,
        maxchunks=5
    )
    bh_output = {}
    for key, content in vbs_out.items():
        bh_output[key] = {
            "hist": content,
            "sumw": sumw_out[key],
    }
    with gzip.open("histogram_%s.pkl.gz" % str(options.jobNum), "wb") as f:
        pickle.dump(bh_output, f)
if __name__ == "__main__":
    main()
