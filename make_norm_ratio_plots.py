import gzip, pickle, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as mh
import os


XSEC_YAML = "/srv/DCTools/config/inc-WZ/xsections_2024.yaml"
LUMI_2024 = 109.08  # /fb

with open(XSEC_YAML) as f:
    XSEC_DB = yaml.safe_load(f)

def dataset_xsec(primary):
    e = XSEC_DB[primary]
    return e["xsec"] * e.get("br", 1.0) * e.get("kr", 1.0)

# alias (matches output/<tagger>/histogram_<alias>.pkl.gz) -> (real DAS primary name, is_signal)
DATASETS = {
    "WZto3LNu_":                     ("WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8", True),
    "ZZJJto2L2Nu_EWK_":              ("ZZJJto2L2Nu-EWK_TuneCP5_13p6TeV_madgraph-pythia8", False),
    "ZZto2L2Nu_":                    ("ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", False),
    "ZZto4L_":                       ("ZZto4L_TuneCP5_13p6TeV_powheg-pythia8", False),
    "ZZZ_5F_":                       ("ZZZ-5F_TuneCP5_13p6TeV_amcatnlo-pythia8", False),
    "WWW_4F_":                       ("WWW-4F_TuneCP5_13p6TeV_amcatnlo-pythia8", False),
    "WWZ_4F_":                       ("WWZ-4F_TuneCP5_13p6TeV_amcatnlo-pythia8", False),
    "WWto2L2Nu_":                    ("WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", False),
    "WWto4Q_":                       ("WWto4Q_TuneCP5_13p6TeV_powheg-pythia8", False),
    "GluGluWWto2Tau2Nu_":            ("GluGluWWto2Tau2Nu_TuneCP5_13p6TeV_mcfm-pythia8", False),
    "GluGluWWtoMuNuTauNu_":          ("GluGluWWtoMuNuTauNu_TuneCP5_13p6TeV_mcfm-pythia8", False),
    "TTto2L2Nu_":                    ("TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", False),
    "TWminusto2L2Nu_":               ("TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", False),
    "DYto2Mu_2Jets_Bin_1J_MLL_50_":  ("DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", False),
    "DYto2Tau_2Jets_Bin_1J_MLL_50_": ("DYto2Tau-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", False),
}

def load_normalized_hist(tagger, alias, primary, variable, channel, systematic="nominal", template=None):
    path = f"output/{tagger}/histogram_{alias}.pkl.gz"
    if not os.path.exists(path):
        return None  # dataset was never successfully produced -- not an error, just absent
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    inner = data[primary]
    hvar = inner["hist"][variable]
    try:
        h = hvar[{"channel": channel, "systematic": systematic}]
    except KeyError:
        print(f"  NOTE: '{channel}' has zero survivors for {alias} ({tagger}) -- treating as empty")
        h = template.copy()
        h.reset()
    norm = dataset_xsec(primary) * 1000.0 * LUMI_2024 / inner["sumw"]
    return h * norm

def get_signal_and_background(tagger, variable, channel):
    template_primary = DATASETS["WZto3LNu_"][0]
    with gzip.open(f"output/{tagger}/histogram_WZto3LNu_.pkl.gz", "rb") as f:
        template_data = pickle.load(f)
    template = template_data[template_primary]["hist"][variable][{"channel": channel, "systematic": "nominal"}]

    h_signal, h_bkg_total = None, None
    for alias, (primary, is_signal) in DATASETS.items():
        h = load_normalized_hist(tagger, alias, primary, variable, channel, template=template)
        if h is None:
            print(f"  SKIP: {alias} ({tagger}) -- no output file produced")
            continue
        if is_signal:
            h_signal = h
        else:
            h_bkg_total = h if h_bkg_total is None else h_bkg_total + h
    return h_signal, h_bkg_total

def make_ratio_plots(variable, channel, outname):
    s_deeptau, b_deeptau = get_signal_and_background("deeptau", variable, channel)
    s_pnet,    b_pnet    = get_signal_and_background("pnet",    variable, channel)
    s_upart,   b_upart   = get_signal_and_background("upart",   variable, channel)

    combos = [
        ("pnet_signal",      s_pnet,  s_deeptau, "PNet"),
        ("upart_signal",     s_upart, s_deeptau, "UParT"),
        ("pnet_background",  b_pnet,  b_deeptau, "PNet"),
        ("upart_background", b_upart, b_deeptau, "UParT"),
    ]
    for label, h_new, h_ref, name in combos:
        if h_new is None or h_ref is None:
            print(f"  SKIP: {label} -- one or both histograms are missing")
            continue
        fig, ax_main, ax_comp = mh.comp.hists(
            h_new, h_ref,
            comparison="ratio",
            xlabel=variable, ylabel="Events (normalized)",
            h1_label=name, h2_label="DeepTau",
        )
        fig.savefig(f"{outname}_{label}.png")
        plt.close(fig)

for var in ("tau_pt_vtight", "met_pt", "dilep_m", "nhtaus_vtight"):
    make_ratio_plots(var, "inc-SR0", f"ratio_norm_{var}")

print("done")
