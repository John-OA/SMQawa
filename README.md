# Installation
git clone this project into a folder which will include several other repos, then execute the bootstrap file with arguments <shell> and <location>.
<shell> is one of `zsh` or `bash`, matching your default shell
<location> is one of `lpc` or `lxplus` and must match where you're working from to setup bind paths correctly

## One-time Setup
Perform initial git clone, and use bootstrap.zsh to create shell and (.zshrc or .bashrc) profiles.
The shell executable sets up environment variables and sources the singularity image (with a default value that can be overridden with a different version for quick testing)
The profiles implicitly create the python virtual environment when none is already present (`.env`), then install coffea and SMQawa in editable mode, and otherwise source the environment if `.env` exists (so for reinstallation of code, you may need to delete the `.env` to trigger a reinstall via the profile).

```bash
export INSTALL_LOC_EXTERNAL=$PWD/WZAnalysis
mkdir -p $INSTALL_LOC_EXTERNAL
cd $INSTALL_LOC_EXTERNAL
git clone -b <branch_name> git@github.com:<githubusername>/SMQawa.git
zsh SMQawa/bootstrap.zsh <shell> <location>
```

## Every-time setup
Navigate to the installation location (where shell and .*rc files are created, parent folder of SMQawa) and run
```bash
./shell
```

## Executing SMQawa code locally
```bash
cd SMQawa
python brewer-remote-inclusive.py --isMC=1 --era=2018 --infile=/store/mc/RunIISummer20UL16NanoAODAPVv9/DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/130000/9FBF4AF1-D77E-4648-916B-097401C9544B.root
```
