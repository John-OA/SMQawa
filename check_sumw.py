import gzip, pickle

taggers = ["deeptau", "pnet", "upart"]
datasets = {
    "WZto3LNu":        "WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8",
    "DYto2E_0J":       "DYto2E-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
    "DYto2E_1J":       "DYto2E-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
    "DYto2E_2J":       "DYto2E-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
    "DYto2Mu_0J":      "DYto2Mu-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
    "DYto2Mu_1J":      "DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
    "DYto2Mu_2J":      "DYto2Mu-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
}

loaded = {}
for tagger in taggers:
    path = f"/srv/SMQawa/final_merged_hists/merged-histogram-{tagger}-mc-2024.pkl.gz"
    with gzip.open(path, "rb") as f:
        loaded[tagger] = pickle.load(f)

print(f"{'dataset':12s} {'deeptau sumw':>18s} {'pnet sumw':>18s} {'upart sumw':>18s}   match?")
for label, primary in datasets.items():
    sumws = {}
    for tagger in taggers:
        try:
            sumws[tagger] = loaded[tagger][primary]["sumw"]
        except KeyError:
            sumws[tagger] = None
    vals = [sumws[t] for t in taggers]
    ok = all(v is not None for v in vals) and max(vals) - min(vals) == 0
    close = all(v is not None for v in vals) and (max(vals) - min(vals)) / max(vals) < 1e-9 if ok is False and all(v is not None for v in vals) else ok
    flag = "EXACT MATCH" if ok else ("close (rel diff < 1e-9)" if close else "MISMATCH")
    print(f"{label:12s} {str(sumws['deeptau']):>18s} {str(sumws['pnet']):>18s} {str(sumws['upart']):>18s}   {flag}")