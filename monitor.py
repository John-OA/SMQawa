import os
import argparse
import shutil
import logging
import subprocess
import rich
from pandas.core.internals.array_manager import new_block
# from termcolor import colored
import importlib.metadata
qawa_version = "0.0.7"

logging.basicConfig(level=logging.INFO)

rerun_script_header = f"""#!/bin/bash
# cd /srv/
# python -m venv --without-pip --system-site-packages jobenv
# source jobenv/bin/activate
# python -m pip install scipy --upgrade --no-cache-dir
# python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-{qawa_version}-py2.py3-none-any.whl

echo "... start job at" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- directory before running:"
ls -lthr
"""


resub_script_header = """#!/bin/bash
# export X509_USER_PROXY={proxy}

# python -m venv --without-pip --system-site-packages jobenv
# source jobenv/bin/activate
# python -m pip install scipy --upgrade --no-cache-dir
# python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-{qawa_version}-py2.py3-none-any.whl

# echo "... start job at" `date "+%Y-%m-%d %H:%M:%S"`
# echo "----- directory before running:"
# echo "----- Found Proxy in: $X509_USER_PROXY"
export X509_USER_PROXY={proxy}
export XRD_REQUESTTIMEOUT=6400
export XRD_REDIRECTLIMIT=64
export INSTALL_LOC_EXTERNAL={install_loc_external}
export COFFEA_IMAGE={coffea_image}
export FULL_IMAGE={full_image}

voms-proxy-info -all
voms-proxy-info -all -file {proxy}

echo "----- COFFEA_IMAGE :"
echo COFFEA_IMAGE $COFFEA_IMAGE
echo FULL_IMAGE $FULL_IMAGE

echo "----- Sourcing virtual environment :"
echo source $INSTALL_LOC_EXTERNAL/.env/bin/activate
source $INSTALL_LOC_EXTERNAL/.env/bin/activate
echo "which python3"
which python3
echo '$SHELL'
echo $SHELL
echo '$PYTHONPATH'
echo $PYTHONPATH
echo '$PYTHON3PATH'
echo $PYTHON3PATH
echo '$PYTHONHOME'
echo $PYTHONHOME
echo '$PATH'
echo $PATH
echo awkward, uproot, coffea, qawa versions:
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show awkward
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show uproot
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show coffea
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show qawa
echo import and print __file__ for coffea
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -c "import coffea; print(coffea.__version__); print(coffea.__file__)"
echo import and print __file__ for qawa
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -c "import qawa; print(qawa.__file__)"


echo "----- JOB STARTS @" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- X509_USER_PROXY    : $X509_USER_PROXY"
echo "----- XRD_REDIRECTLIMIT  : $XRD_REDIRECTLIMIT"
echo "----- XRD_REQUESTTIMEOUT : $XRD_REQUESTTIMEOUT"
ls -lthr
{command}

echo "----- directory after running :"
ls -lthr
if [ ! -f "histogram_$1.pkl.gz" ]; then
  echo "No output histogram pickle file found";
  exit 1;
fi
echo " ------ THE END (everyone dies !) ----- "
"""

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-a"   , "--analysis", type=str, default="inc-WZ"       , help="Analysis to run", required=True)
    parser.add_argument("-i"   , "--input" , type=str, default="input"  , required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="algiers", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1        , help="")
    parser.add_argument("-e"   , "--era"   , type=str, default="2018"   , help="")
    parser.add_argument("--runlocal", action="store_true")
    parser.add_argument("--resubmit", action="store_true", help="resubmit failed jobs")
    parser.add_argument("--dryrun"  , action="store_true")
    parser.add_argument('--executor' , type=str, default="FuturesExecutor", help="Executor to use, one of IterativeExecutor (good for debugging), FuturesExecutor (multithreaded), or other coffea option")
    parser.add_argument('--copyInput', action='store_true'     , help="xrdcp a file to the worker node before executing the coffea processor on it")
    parser.add_argument('--verbose'  , action='store_true'     , help="verbose output printing status of running, finished, and failed files per job")
    options = parser.parse_args()


    captured_env = os.environ.copy()
    if "bash" in captured_env['SHELL']:
        to_source = os.path.join(captured_env['INSTALL_LOC'], ".bashrc")
    elif "zsh" in captured_env['SHELL']:
        to_source = os.path.join(captured_env['INSTALL_LOC'], ".zshrc")
    else:
        raise NotImplementedError("neither bash or zsh detected in the shell env variable, something has gone wrong; contents=", captured_env['SHELL'])

    # condor_status = os.popen('condor_q -nobatch').read()
    condor_stat_cmd = f"cd {captured_env['INSTALL_LOC']} && source {to_source} && condor_q -nobatch"
    # condor_stat_cmd = f"cd {captured_env['INSTALL_LOC']} && condor_q -nobatch"
    logging.info(f"condor command : {condor_stat_cmd}")
    htc = subprocess.Popen(
        condor_stat_cmd,
        shell      = True,
        executable = captured_env['SHELL'],
        env        = captured_env,
        stdin      = subprocess.PIPE,
        stdout     = subprocess.PIPE,
        stderr     = subprocess.PIPE,
        close_fds  = True
    )
    condor_status, htc_err = htc.communicate()
    condor_status = str(condor_status) # need to convert bytes object to string
    exit_status = htc.returncode
    logging.info(f"condor q -nobatch status : {exit_status}")
    # logging.info(f"condor q -nobatch stdout : {condor_status}")
    logging.info(f"condor q -nobatch stderr : {htc_err}")
    # condor_status = os.popen(condor_stat_cmd).read()
    print("condor_status:", condor_status)
    # condor_status = os.popen('condor_q -nobatch').read()
   
    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    user_name  = os.environ['USER']
    proxy_copy = os.path.join(home_base,proxy_base)
    coffea_image = os.environ['COFFEA_IMAGE']
    full_image = os.environ['FULL_IMAGE']
    install_loc_external = os.environ['INSTALL_LOC_EXTERNAL']
    brewer_loc_external = os.path.join(os.environ['INSTALL_LOC_EXTERNAL'], "SMQawa", "brewer-remote-inclusive.py")
    
    if not os.path.isfile(proxy_copy):
        logging.warning('--- proxy file does not exist')
    else:
        lifetime = subprocess.check_output(
            ['voms-proxy-info', '--file', proxy_copy, '--timeleft']
        )    
        lifetime = float(lifetime)
        lifetime = lifetime / (60*60)
        logging.info("--- proxy lifetime is {} hours".format(lifetime))
        if lifetime < 10.0: # we want at least 10 hours
            logging.warning("--- proxy has expired !")


    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            if len(sample.split('/')) <= 1: continue
            sample_name = sample.split("/")[1] if options.isMC else '_'.join(sample.split("/")[1:3])
            jobs_dir = '_'.join(['jobs', options.tag, options.era, sample_name])
            jobs_dir_external = os.path.join(os.environ['INSTALL_LOC_EXTERNAL'], os.path.relpath(os.path.normpath(jobs_dir), os.environ['INSTALL_LOC']))

            input_root_files = list(open(jobs_dir + "/" + "inputfiles.dat").read().splitlines())
            
            n_jobs = len(input_root_files)

            job_running = []
            job_failed = []
            job_finished = []
            resubmit_list = {}
            for idf, rfn in enumerate(input_root_files):
                if rfn in str(condor_status):
                    # print(f"Debug found job in running status: {idf} -  {rfn}")
                    if options.verbose:
                        rich.print(f"Debug found job in [green]running status: {idf} -  {rfn}[/green]")
                    job_running.append(rfn)
                elif os.path.exists(jobs_dir_external + f'/histogram_{idf}.pkl.gz'):
                    # print(f"Debug found histogram: {idf} -  {rfn} - {jobs_dir_external}/histogram_{idf}.pkl.gz")
                    if options.verbose:
                        rich.print(f"Debug [blue]found histogram: {idf} -  {rfn} - {jobs_dir_external}/histogram_{idf}.pkl.gz[/blue]")
                    job_finished.append(rfn)
                else:
                    # print(f"Debug classifying job as failed: {idf} -  {rfn}")
                    if options.verbose:
                        rich.print(f"Debug [red]classifying job as failed: {idf} -  {rfn}[/red]")
                    job_failed.append(rfn)
                    resubmit_list[idf] = rfn
            logging.info(
                "-- {:62s}".format((sample_name[:60] + '..') if len(sample_name)>60 else sample_name) +
                (
                    f" --> {n_jobs:5d} : completed" if n_jobs==len(job_finished) else 
                    f" --> {n_jobs:5d} : {len(job_running):5d} {n_jobs-len(job_failed)-len(job_running):5d} {len(job_failed):5d}"
                    # colored(f" --> {n_jobs:5d} : completed", "green") if n_jobs==len(job_finished) else colored(
                    #     f" --> {n_jobs:5d} : {len(job_running):5d}", 'yellow'
                    # )+colored(
                    #     f"{n_jobs-len(job_failed)-len(job_running):5d}", "green"
                    # )+colored(
                    #     f"{len(job_failed):5d}", 'red'
                    # )
                )
            )

            if len(job_running)>0:
                for rfn in job_running:
                    # logging.debug(colored(f'running : {rfn}', 'yellow'))
                    logging.debug(f'running : {rfn}')
            if len(job_failed)>0:
                for rfn in job_failed:
                    # logging.debug(colored(f'failed  : {rfn}', 'red'))
                    logging.debug(f'failed  : {rfn}')
            
            if options.resubmit and len(job_failed)>0: 
                # shutil.copyfile('brewer-remote-inclusive.py', jobs_dir+'/brewer-remote-inclusive.py') #FIXME: still needed?
                local_rerun_lines = [rerun_script_header]
                for jid,infile in resubmit_list.items():
                    infile_name = infile 
                    split_args = infile.split('/')
                    auto_isMC = None  #if neither data or mc tag is found, keep as None
                    auto_dataset = None
                    auto_runperiod = ""
                    if "NANOAODSIM" in split_args:
                        auto_isMC = True
                    elif "NANOAOD" in split_args:
                        auto_isMC = False
                    else:
                        pass
                    if auto_isMC is not None:
                        assert ((options.isMC==1) == auto_isMC), f"auto MC detection is not consistent with isMC command line option: (isMC==1)={options.isMC==1} :: auto_isMC={auto_isMC}"
                        try:
                            tier_index = split_args.index("NANOAODSIM" if auto_isMC else "NANOAOD")
                            auto_dataset = split_args[tier_index - 1]
                            auto_runperiod = split_args[tier_index - 2].replace(f"Run{options.era}", "")
                        except ValueError as ve:
                            print("couldn't auto-parse dataset and runperiod from filename:)")
                            print(ve)
                    run_period = auto_runperiod
                    dataset_name = auto_dataset
                    # dataset_name = infile.split('/')[4]
                    # run_period = ''
                    # if options.isMC:
                    #     run_period = ''
                    # else:
                    #     run_period = infile.split('/')[3].replace(f'Run{options.era}','')
                    if options.runlocal:
                        assert options.era != "", f"please specify the era you are rerunning ... example: --era=2018"
                        # options.copyfile now built into brewer-remote-inclusive.py with --copyFile command
                        # if options.copyfile:
                        #     local_rerun_lines.append(
                        #         "xrdcp {infile} ."
                        #     )
                        #     infile_name = infile.split('/')[-1]
                        local_rerun_lines.append(
                            f"$INSTALL_LOC_EXTERNAL/.env/bin/python3 brewer-remote-inclusive.py --jobNum={jid} --isMC={options.isMC} --era={options.era} --infile={infile_name} --executor={options.executor} {'--copyInput' if options.copyInput else ''}\n"
                            # f"python brewer-remote-inclusive.py --jobNum={jid} --isMC={options.isMC} --era={options.era} --infile={infile_name} --dataset={dataset_name}\n"
                        )
                        if options.copyInput:
                            local_rerun_lines[-1].replace("\n", " --copyInput\n")
                    else:
                        if options.copyInput:
                            assert options.era != "", f'please specify the era of the dataset you are running. ex: --era=2018'
                            # script_command = f"xrdcp root://cms-xrd-global.cern.ch/$2 . \n"
                            # infile_name = infile.split('/')[-1]
                            # script_command += f"python3 brewer-remote-inclusive.py --jobNum=$1 --isMC={options.isMC} --era={options.era} --infile={infile_name} --dataset={dataset_name} --runperiod={run_period} --copyInput\n"
                            script_command = f"$INSTALL_LOC_EXTERNAL/.env/bin/python3 brewer-remote-inclusive.py --jobNum=$1 --isMC={options.isMC} --era={options.era} --infile={infile_name} --dataset={dataset_name} --runperiod={run_period}--executor={options.executor} {'--copyInput' if options.copyInput else ''}\n"
                            # script_command += f"rm {infile_name}\n" Should not be needed, file is downloaded to a temporary directory which gets cleaned
                            script_command += "ls -lthr\n"
                            with open(os.path.join(jobs_dir, f"resub-script-{jid}.sh"), "w") as _stream:
                                script_file_ = resub_script_header.format(
                                    proxy=proxy_copy, 
                                    qawa_version=qawa_version,
                                    coffea_image=coffea_image,
                                    full_image=full_image,
                                    install_loc_external=install_loc_external,
                                    command=script_command,
                                    jobid=jid,
                                )
                                _stream.write(script_file_)
                                _stream.close()
                        else: 
                            assert options.era != "", f'please specify the era of the dataset you are running. ex: --era=2018'
                            script_command = f"$INSTALL_LOC_EXTERNAL/.env/bin/python3 brewer-remote-inclusive.py --jobNum=$1 --isMC={options.isMC} --era={options.era} --infile=$2 --executor={options.executor} {'--copyInput' if options.copyInput else ''}\n"
                            # script_command = f"python brewer-remote-inclusive.py --jobNum=$1 --isMC={options.isMC} --era={options.era} --infile=$2\n"
                            script_command += "ls -lthr\n"
                            with open(os.path.join(jobs_dir, f"resub-script-{jid}.sh"), "w") as _stream:
                                script_file_ = resub_script_header.format(
                                    proxy=proxy_copy, 
                                    qawa_version=qawa_version,
                                    coffea_image=coffea_image,
                                    full_image=full_image,
                                    install_loc_external=install_loc_external,
                                    command=script_command,
                                    jobid=jid
                                )
                                _stream.write(script_file_)
                                _stream.close()

                        condor_sub = open(jobs_dir + "/condor.sub").readlines()
                        for il, line in enumerate(condor_sub):
                            if 'executable' in line.lower(): 
                                condor_sub [il] = f'executable = {jobs_dir_external}/resub-script-{jid}.sh\n'
                                #condor_sub [il] = f'executable = {jobs_dir}/script.sh\n'
                            if 'arguments' in line.lower():
                                condor_sub [il] = f"arguments = {jid} {infile}\n"
                            if 'jobflavour' in line.lower():
                                condor_sub [il] = '+JobFlavour           = "workday"\n'
                            if 'queue' in line.lower():
                                condor_sub[il] = "queue"
                        with open(jobs_dir + f'/condor_resub_{jid}.sub', 'w') as new_condor:
                            new_condor.writelines(condor_sub)
                            new_condor.close()
                        
                        if not options.dryrun:
                            cmd = f"cd {captured_env['INSTALL_LOC']} && source {to_source} && condor_submit {os.path.join(jobs_dir_external, f'condor_resub_{jid}.sub')}"
                            # htc = os.popen("condor_submit " + os.path.join(jobs_dir, f"condor_resub_{jid}.sub")).read()
                            htc = subprocess.Popen(
                                cmd,
                                shell      = True,
                                executable = captured_env['SHELL'],
                                env        = captured_env,
                                stdin      = subprocess.PIPE,
                                stdout     = subprocess.PIPE,
                                stderr     = subprocess.PIPE,
                                close_fds  =True
                            )

                            htc_out, htc_err = htc.communicate()
                            exit_status = htc.returncode
                            logging.info(f"condor submission status : {exit_status}")
                            logging.info(f"condor communicate stdout : {htc_out}")
                            logging.info(f"condor communicate stderr : {htc_err}")
                            # logging.info(htc)
                
                if options.runlocal:
                    raise NotImplementedError("runlocal has not been modified to run inside the container yet, properly referencing the bash/zsh-shell bootstrap and environment variables pointing to the image used 'FULL_IMAGE'")
                    with open(os.path.join(jobs_dir, f"rerun-script.sh"), "w") as _stream:
                        _stream.writelines(local_rerun_lines)

                    coffea_image = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest" 
                    os.system(f"cp dist/Qawa-0.0.7-py2.py3-none-any.whl {jobs_dir}")   
                    if not options.dryrun:
                        htc = os.popen(f"singularity exec -B {jobs_dir}:/srv/ {coffea_image} bash /srv/rerun-script.sh").read()
                        print(htc)
                    
                    


if __name__ == "__main__":
    main()

