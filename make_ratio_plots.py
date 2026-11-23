import gzip, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as mh

DATASET = "WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8"

def load_hist(tagger, variable, channel, systematic="nominal"):
    with gzip.open(f"histogram_{tagger}.pkl.gz", "rb") as f:
        data = pickle.load(f)
    h = data[DATASET]["hist"][variable]
    return h[{"channel": channel, "systematic": systematic}]

def make_ratio_plot(variable, channel, outname):
    h_deeptau = load_hist("deeptau", variable, channel)
    h_pnet    = load_hist("pnet",    variable, channel)
    h_upart   = load_hist("upart",   variable, channel)

    fig, ax_main, ax_comp = mh.comp.hists(
        h_pnet, h_deeptau,
        comparison="ratio",
        xlabel=variable, ylabel="Events",
        h1_label="PNet", h2_label="DeepTau",
    )
    fig.savefig(f"{outname}_pnet_vs_deeptau.png")
    plt.close(fig)

    fig, ax_main, ax_comp = mh.comp.hists(
        h_upart, h_deeptau,
        comparison="ratio",
        xlabel=variable, ylabel="Events",
        h1_label="UParT", h2_label="DeepTau",
    )
    fig.savefig(f"{outname}_upart_vs_deeptau.png")
    plt.close(fig)

for var in ("tau_pt_vtight", "met_pt", "dilep_m", "nhtaus_vtight"):
    make_ratio_plot(var, "inc-SR0", f"ratio_{var}")

print("done")
