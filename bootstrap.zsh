#!/usr/bin/env zsh

if [[ "$1" == "zsh" ]]; then
    cat <<EOF > shell
#!/usr/bin/env zsh

export INSTALL_LOC_EXTERNAL=\$PWD
export INSTALL_LOC=/srv/
export ZDOTDIR=\$INSTALL_LOC

EOF
else
    cat <<EOF > shell
#!/usr/bin/env bash

export INSTALL_LOC_EXTERNAL=\$PWD
export INSTALL_LOC=/srv/

EOF
fi

if [[ "$2" == "lpc" ]]; then
    cat <<EOF >> shell
# Needed to setup cluster for LPC
export CONDOR_CONFIG=\$INSTALL_LOC.condor_config
grep -v '^include' /etc/condor/config.d/01_cmslpc_interactive > .condor_config

# Need all our bind addresses
export APPTAINER_BINDPATH=/uscmst1b_scratch,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/eos,/etc/pki/ca-trust,/run/user,/var/run/user

EOF
else
    cat <<EOF >> shell
# Need all our bind addresses
export APPTAINER_BINDPATH=/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/eos,/etc/pki/ca-trust,/etc/tnsnames.ora,/run/user,/var/run/user

EOF
fi

cat <<EOF >> shell
voms-proxy-init -voms cms --valid 192:00 --out \$HOME/x509up_u\$UID
export X509_USER_PROXY=\$HOME/x509up_u\$UID

if [[ "\$1" == "" ]]; then
  export COFFEA_IMAGE="coffeateam/coffea-base-almalinux9:0.7.26-py3.10"
else
  export COFFEA_IMAGE="\$1"
fi

export FULL_IMAGE="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/"\$COFFEA_IMAGE
EOF

if [[ "$1" == "zsh" ]]; then
    cat <<EOF >> shell
SINGULARITY_SHELL=\$(which zsh) singularity exec -B \${PWD}:/srv --pwd /srv \${FULL_IMAGE} $(which zsh)
EOF
else
    cat <<EOF >> shell
SINGULARITY_SHELL=\$(which bash) singularity exec -B \${PWD}:/srv --pwd /srv \${FULL_IMAGE} $(which bash) --rcfile /srv/.bashrc
EOF
fi

if [[ "$1" == "zsh" ]]; then
    cat <<EOF > .zshrc
export XRDPARALLELEVTLOOP=16 #This might only work in development environments, but should increase the throughput...
# export INSTALL_LOC=\$PWD/ #this could potentially be VIRTUAL_ENV, but creating that prior to activation may cause unforeseen problems... note TRAILING slash
if [[ -z "\$INSTALL_LOC" ]]; then
  echo "INSTALL_LOC not set, check shell script or export the variable for where the virtual environment should be installed/found"
else
  echo "INSTALL_LOC=" \$INSTALL_LOC
fi

test -e \${ZDOTDIR}/.iterm2_shell_integration.zsh && source \${ZDOTDIR}/.iterm2_shell_integration.zsh


install_env() {
  print "INSTALLING ENV"
  # This will break if the repo isn't cloned first
  set -e
  echo "Installing shallow virtual environment in \$PWD/.env..."
  python -m venv --without-pip --system-site-packages \$INSTALL_LOC.env
  unlink \$INSTALL_LOC.env/lib64  # HTCondor can't transfer symlink to directory and it appears optional
  cd \${INSTALL_LOC}
  git clone -b smqawa-zz2l2nu http://github.com/NJManganelli/coffea.git
  cd coffea
  \$INSTALL_LOC.env/bin/python -m pip install -e .
  cd ..
  if [ ! -d "SMQawa" ]; then
    echo "SMQawa must already be cloned, e.g. via 'git clone -b <branch> git@github.com:<githubusername>/SMQawa.git'"
    echo "the bootstrap.zsh script should be run from the parent folder of SMQawa to allow editable install of coffea and other packages alongside it."
    echo "clean the virtual env before re-attempting install."
  fi
  cd SMQawa
  \$INSTALL_LOC.env/bin/python -m pip install -e .
  # mkdir -p build
  # \$INSTALL_LOC.env/bin/python -m pip wheel --no-deps .
  # \$INSTALL_LOC.env/bin/python -m pip install Qawa*.whl
  cd ..
  echo "done."
}

install_kernel() {
  # work around issues copying CVMFS xattr when copying to tmpdir
  export TMPDIR=\$(mktemp -d -p .)
  \$INSTALL_LOC.env/bin/python -m ipykernel install --user --name smqawa --display-name "smqawa" --env PYTHONPATH $PYTHONPATH:$PWD --env PYTHONNOUSERSITE 1
  rm -rf \$TMPDIR && unset TMPDIR
}

install_all() {
  install_env
  install_kernel
  # source \$INSTALL_LOC.env/bin/activate
}

export JUPYTER_PATH=\$INSTALL_LOC.jupyter
export JUPYTER_RUNTIME_DIR=\$INSTALL_LOC.local/share/jupyter/runtime
export JUPYTER_DATA_DIR=\$INSTALL_LOC.local/share/jupyter
export IPYTHONDIR=\$INSTALL_LOC.ipython
unset GREP_OPTIONS

[[ -d \$INSTALL_LOC.env ]] || install_all
source \$INSTALL_LOC.env/bin/activate
alias pip="python -m pip"

EOF
    curl -L https://iterm2.com/shell_integration/zsh -o ~/.iterm2_shell_integration.zsh
    chmod u+x shell .zshrc
    echo "Wrote shell and .zshrc to current directory. Run ./shell to start the singularity shell"
else
    cat <<EOF > .bashrc
export XRDPARALLELEVTLOOP=16 #This might only work in development environments, but should increase the throughput...
# export INSTALL_LOC=\$PWD/ #this could potentially be VIRTUAL_ENV, but creating that prior to activation may cause unforeseen problems... note TRAILING slash
if [[ -z "\$INSTALL_LOC" ]]; then
  echo "INSTALL_LOC not set, check shell script or export the variable for where the virtual environment should be installed/found"
else
  echo "INSTALL_LOC=" \$INSTALL_LOC
fi

install_env() {
  # This will break if the repo isn't cloned first
  set -e
  echo "Installing shallow virtual environment in \$INSTALL_LOC.env..."
  python -m venv --without-pip --system-site-packages \$INSTALL_LOC.env
  unlink \$INSTALL_LOC.env/lib64  # HTCondor can't transfer symlink to directory and it appears optional
  cd \${INSTALL_LOC}
  git clone -b smqawa-zz2l2nu http://github.com/NJManganelli/coffea.git
  cd coffea
  \$INSTALL_LOC.env/bin/python -m pip install -e .
  cd ..
  if [ ! -d "SMQawa" ]; then
    echo "SMQawa must already be cloned, e.g. via 'git clone -b <branch> git@github.com:<githubusername>/SMQawa.git'"
    echo "the bootstrap.zsh script should be run from the parent folder of SMQawa to allow editable install of coffea and other packages alongside it."
    echo "clean the virtual env before re-attempting install."
  fi
  cd SMQawa
  \$INSTALL_LOC.env/bin/python -m pip install -e .
  # mkdir -p build
  # \$INSTALL_LOC.env/bin/python -m pip wheel --no-deps .
  # \$INSTALL_LOC.env/bin/python -m pip install Qawa*.whl
  cd ..
  echo "done."
}

install_kernel() {
  # work around issues copying CVMFS xattr when copying to tmpdir
  export TMPDIR=\$(mktemp -d -p .)
  \$INSTALL_LOC.env/bin/python -m ipykernel install --user --name smqawa --display-name "smqawa" --env PYTHONPATH $PYTHONPATH:$PWD --env PYTHONNOUSERSITE 1
  rm -rf \$TMPDIR && unset TMPDIR
}

install_all() {
  install_env
  install_kernel
  # source \$INSTALL_LOC.env/bin/activate
}

export JUPYTER_PATH=\$INSTALL_LOC.jupyter
export JUPYTER_RUNTIME_DIR=\$INSTALL_LOC.local/share/jupyter/runtime
export JUPYTER_DATA_DIR=\$INSTALL_LOC.local/share/jupyter
export IPYTHONDIR=\$INSTALL_LOC.ipython
unset GREP_OPTIONS

[[ -d .env ]] || install_all
source .env/bin/activate
alias pip="python -m pip"

EOF
    chmod u+x shell .bashrc
    echo "Wrote shell and .bashrc to current directory. Run ./shell to start the singularity shell"
fi
