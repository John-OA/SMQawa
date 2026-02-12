import gzip
import pickle
import argparse
import collections
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def load_and_process_file(filename):
    if os.path.getsize(filename) == 0:
        return None

    try:
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)

        result = {}
        for s, v in data.items():
            # Flatten hist (skip nested dicts)
            flat_hist = {k: v_ for k, v_ in v['hist'].items() if not isinstance(v_, dict)}
            sumw = v['sumw']
            result[s] = (flat_hist, sumw)

        return result

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def merger():
    parser = argparse.ArgumentParser(description='Fast Histogram Merger')
    parser.add_argument("-t", '--tag', type=str, default="algiers", help="Tag name to match files")
    parser.add_argument('--era', type=str, default="2018", help="Era to match files")
    parser.add_argument("-f",'--filename', type=str, default="latest_DD", help="file name")
    options = parser.parse_args()

    file_list = glob.glob(f'*{options.tag}*_{options.era}_*/*.pkl.gz')

    combined_hist = {}
    combined_sumw = {}

    # Use all available CPU cores
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(load_and_process_file, file_list), total=len(file_list), desc="Processing", ncols=75):
            if result is None:
                continue

            for s, (hist, sumw) in result.items():
                if s not in combined_hist:
                    combined_hist[s] = collections.Counter()
                    combined_sumw[s] = 0.0
                combined_hist[s].update(hist)
                combined_sumw[s] += sumw

    combined_dict = {
        s: {"hist": dict(h), "sumw": combined_sumw[s]}
        for s, h in tqdm(combined_hist.items(), desc="Finalizing", ncols=75)
    }

    output_file = f"merged-histogram-Inc-{options.tag}-{options.era}-{options.filename}.pkl.gz"
    with gzip.open(output_file, "wb") as f:
        pickle.dump(combined_dict, f)

    print(f"Done. Output written to: {output_file}")

if __name__ == "__main__":
    merger()
