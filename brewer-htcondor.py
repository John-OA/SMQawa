import os
import argparse
import logging
import subprocess
import shutil
import time
import importlib.metadata

logging.basicConfig(level=logging.DEBUG)

# qawa_version = importlib.metadata.version('qawa')
qawa_version = '0.0.7'

script_TEMPLATE_data = """#!/bin/bash
export X509_USER_PROXY={proxy}
export XRD_REQUESTTIMEOUT=6400
export XRD_REDIRECTLIMIT=64
export PYTHONPATH=/afs/cern.ch/user/h/hgao/.local/lib/python3.10/site-packages:$PYTHONPATH

voms-proxy-info -all
voms-proxy-info -all -file {proxy}


python -m venv --without-pip --system-site-packages jobenv
source jobenv/bin/activate
python -m pip install scipy --upgrade --no-cache-dir
python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-{qawa_version}-py2.py3-none-any.whl

echo "----- JOB STARTS @" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- X509_USER_PROXY    : $X509_USER_PROXY"
echo "----- XRD_REDIRECTLIMIT  : $XRD_REDIRECTLIMIT"
echo "----- XRD_REQUESTTIMEOUT : $XRD_REQUESTTIMEOUT"
ls -lthr
echo "----- this is a data file"
filepath=$2
echo "----- processing the files : "
dataset=$(echo "$filepath" | awk -F'/{era}/' '{{print $2}}' | awk -F'/' '{{print $1}}' | awk -F'_' '{{print $1}}')
runperiod=$(echo "$filepath" | awk -F'/{era}/' '{{print $2}}' | awk -F'_' '{{split($2, a, "-"); print substr(a[1], length(a[1]), 1)}}')

mkdir -p /tmp/hgao/$dataset
a=$(basename $2)
cp $2 /tmp/hgao/$dataset
ls -lthr /tmp/hgao/$dataset/
python brewer-remote.py --jobNum=$1 --isMC={ismc} --era={era} --dd={dd} --infile=/tmp/hgao/$dataset/$a --dataset=$dataset --runperiod=$runperiod 
rm /tmp/hgao/$dataset/*
echo "----- directory after running :"
ls -lthr
if [ ! -f "histogram_$1.pkl.gz" ]; then
  exit 1;
fi
echo " ------ THE END (everyone dies !) ----- "
"""

script_TEMPLATE_MC = """#!/bin/bash
export X509_USER_PROXY={proxy}
export XRD_REQUESTTIMEOUT=6400
export XRD_REDIRECTLIMIT=64
export PYTHONPATH=/afs/cern.ch/user/h/hgao/.local/lib/python3.10/site-packages:$PYTHONPATH

voms-proxy-info -all
voms-proxy-info -all -file {proxy}

python -m venv --without-pip --system-site-packages jobenv
source jobenv/bin/activate
python -m pip install scipy --upgrade --no-cache-dir
python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-{qawa_version}-py2.py3-none-any.whl

echo "----- JOB STARTS @" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- X509_USER_PROXY    : $X509_USER_PROXY"
echo "----- XRD_REDIRECTLIMIT  : $XRD_REDIRECTLIMIT"
echo "----- XRD_REQUESTTIMEOUT : $XRD_REQUESTTIMEOUT"
ls -lthr
echo "----- this is a MC file"
filepath=$2
echo "----- processing the files : "
dataset=$(echo "$filepath" | awk -F'/{era}/' '{{print $2}}' | awk -F'/' '{{print $1}}')

mkdir -p /tmp/hgao/$dataset
a=$(basename $2)
cp $2 /tmp/hgao/$dataset
ls -lthr /tmp/hgao/$dataset/
python brewer-remote.py --jobNum=$1 --isMC={ismc} --era={era} --dd={dd} --infile=/tmp/hgao/$dataset/$a --dataset=$dataset --runperiod=$runperiod
rm /tmp/hgao/$dataset/*

echo "----- directory after running :"
ls -lthr
if [ ! -f "histogram_$1.pkl.gz" ]; then
  exit 1;
fi
echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 1000000000

executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobfn)
transfer_input_files  = {transfer_file}
# transfer_output_files = df_$(ProcId).parquet 
should_transfer_files = YES
WhenToTransferOutput  = ON_EXIT_OR_EVICT
initialdir            = {jobdir}
# output_destination    = root://eosuser.cern.ch//eos/user/y/yhaddad/condor_jobs/{jobdir}/

output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log

on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
max_retries           = 3
request_memory        = 6000M
requirements          = Machine =!= LastRemoteHost
# MY.XRDCP_CREATE_DIR   = True
# +SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:0.7.21-py3.9-gaab39"
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
+JobFlavour           = "{queue}"

queue jobfn from {jobdir}/inputfiles.dat
"""

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt" , help="input datasets", required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="atakour"  , help="production tag", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1          , help="")
    parser.add_argument("-q"   , "--queue" , type=str, default="workday", help="")
    parser.add_argument("-e"   , "--era"   , type=str, default="2018"     , help="")
    parser.add_argument("-f"   , "--force" , action="store_true"          , help="recreate files and jobs")
    parser.add_argument("-s"   , "--submit", action="store_true"          , help="submit only")
    parser.add_argument("-dry" , "--dryrun", action="store_true"          , help="running without submission")
    parser.add_argument("--redo-proxy"     , action="store_true"          , help="redo the voms proxy")
    parser.add_argument("-d"   , "--dd"    , type=str, default="onlySR"     , help="onlySR,DYSR,MC", required=True)
    options = parser.parse_args()

    # Making sure that the proxy is good
    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    user_name  = os.environ['USER']
    proxy_copy = os.path.join(home_base,proxy_base)
    eosbase = f"/eos/user/{user_name[0]}/{user_name}/ZZTo2L2Nu/" + "{tag}/{sample}/"
    input_base =f'/eos/cms/store/group/phys_smp/ZZTo2L2Nu/HZZsample/{options.era}'

    regenerate_proxy = False
    if not os.path.isfile(proxy_copy):
        logging.warning('--- proxy file does not exist')
        regenerate_proxy = True
    else:
        lifetime = subprocess.check_output(
            ['voms-proxy-info', '--file', proxy_copy, '--timeleft']
        )
        print (lifetime)
        lifetime = float(lifetime)
        lifetime = lifetime / (60*60)
        logging.info("--- proxy lifetime is {} hours".format(lifetime))
        if lifetime < 10.0:
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system('voms-proxy-init --valid 120:00 --rfc --voms cms')
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True
        shutil.copyfile('/tmp/'+proxy_base,  proxy_copy)
    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            if len(sample.split('/')) <= 1: continue
            sample_name = sample.split("/")[1] if options.isMC else '_'.join(sample.split("/")[1:3])
            sample_name = sample_name.replace("*", "")
            era_directory = options.era 
            os.makedirs(f"{era_directory}-SR-v2", exist_ok=True)
            jobs_dir_sub = '_'.join(['jobs', options.tag, options.era, sample_name])
            jobs_dir = os.path.join(f"{era_directory}-SR-v2", jobs_dir_sub)
            logging.info("-- sample_name : " + sample)
            if os.path.isdir(jobs_dir):
                if not options.force:
                    logging.error(" " + jobs_dir + " already exist !")
                    continue
                else:
                    logging.warning(" " + jobs_dir + " already exists, forcing its deletion!")
                    shutil.rmtree(jobs_dir)
                    os.mkdir(jobs_dir)
            else:
                os.mkdir(jobs_dir)

            if not options.submit:
                sample_files = []
                if '*' in sample:
                    sample_with_ext = subprocess.check_output(
                        ['ls', f"{input_base}/{sample_name}"]
                    )
                    print(" --- found these samples : ")
                    print(sample_with_ext.decode('UTF-8'))
                    print("and these are the files : ")
                    for sample_ in sample_with_ext.decode("UTF-8").split("\n")[:-1]:
                        output_ = subprocess.check_output(
                            ['ls',f"{input_base}/{sample_}"]
                        )
                        sample_files += list(filter(lambda x: x != '', output_.decode('UTF-8').split('\n')))
                else:
                    output_ = subprocess.check_output(
                        ['ls',f"{input_base}/{sample}"]
                    )
                    sample_files = list(filter(lambda x: x != '', output_.decode('UTF-8').split('\n')))

                time.sleep(1)
                with open(os.path.join(jobs_dir, "inputfiles.dat"), 'w') as infiles:
                    for fn in sample_files:
                        if '.root' in fn:
                            infiles.write(f'{input_base}/{sample_name}/{fn}')
                            infiles.write('\n')
                        else:
                            continue
                    infiles.close()
            time.sleep(2)
            eosoutdir =  eosbase.format(tag=options.tag,sample=sample_name)
            # crete a directory
            os.system("mkdir -p {}".format(eosoutdir))

            with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
                if options.isMC:
                    script = script_TEMPLATE_MC.format(
                        proxy=proxy_copy,
                        ismc=options.isMC,
                        era=options.era,
                        eosdir=eosoutdir, 
                        qawa_version=qawa_version,
                        dd = options.dd
                    )
                else:
                    script = script_TEMPLATE_data.format(
                        proxy=proxy_copy,
                        ismc=options.isMC,
                        era=options.era,
                        eosdir=eosoutdir, 
                        qawa_version=qawa_version,
                        dd = options.dd
                    )                    
                scriptfile.write(script)
                scriptfile.close()

            with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
                condor = condor_TEMPLATE.format(
                    transfer_file= ",".join([
                        f"../../brewer-remote.py",
                        f"../../dist/Qawa-{qawa_version}-py2.py3-none-any.whl",
                    ]),
                    jobdir=jobs_dir,
                    queue=options.queue
                )
                condorfile.write(condor)
                condorfile.close()
            if options.dryrun:
                continue

            htc = subprocess.Popen(
                "condor_submit " + os.path.join(jobs_dir, "condor.sub"),
                shell  = True,
                stdin  = subprocess.PIPE,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                close_fds=True
            )
            
            htc.communicate()
            exit_status = htc.returncode
            logging.info("condor submission status : {}".format(exit_status))
            if exit_status != 0:
                os.system("rm -rf {}".format(jobs_dir))
                if options.isMC:
                    with open(f"./data/datasetUL{options.era}-mc-skim-tmpfail.txt", "a") as failfile:
                        failfile.write(f'/{sample_name}')
                        failfile.write('\n')
                        failfile.close()
                if not options.isMC:
                    with open(f"./data/datasetUL{options.era}-data-skim-tmpfail.txt", "a") as failfile:
                        failfile.write(f'/{sample_name}')
                        failfile.write('\n')
                        failfile.close()
if __name__ == "__main__":
    main()
