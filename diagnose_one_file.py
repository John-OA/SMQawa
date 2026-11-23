import uproot

jobdir = "jobs_2024_deeptau_data_2024_Muon0_Run2024C-MINIv6NANOv15-v1"
with open(f"{jobdir}/inputfiles.dat") as f:
    files = [l.strip() for l in f if l.strip()]

redirector = "root://cms-xrd-global.cern.ch/"
fpath = redirector + files[0]
print("Trying:", fpath)
with uproot.open(fpath + ":Events") as t:
    print("SUCCESS:", t.num_entries, "entries")