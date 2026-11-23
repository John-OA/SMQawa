import uproot
import glob
import os
import sys

REDIRECTOR = "root://cms-xrd-global.cern.ch/"

def count_dir(jobdir):
    inputfile = os.path.join(jobdir, 'inputfiles.dat')
    if not os.path.exists(inputfile):
        return None
    with open(inputfile) as f:
        files = [l.strip() for l in f if l.strip()]
    total = 0
    nfail = 0
    first_error = None
    for fpath in files:
        full_path = fpath if fpath.startswith('root://') else (REDIRECTOR + fpath)
        try:
            with uproot.open(full_path + ':Events') as t:
                total += t.num_entries
        except Exception as e:
            nfail += 1
            if first_error is None:
                first_error = f'{type(e).__name__}: {e}'
    return total, len(files), nfail, first_error

mode = sys.argv[1]  # 'mc' or 'data'
pattern = sys.argv[2]  # glob pattern for job dirs

results = []
for jobdir in sorted(glob.glob(pattern)):
    r = count_dir(jobdir)
    if r is None:
        continue
    total, nfiles, nfail, first_error = r
    name = os.path.basename(jobdir)
    results.append((name, total, nfiles, nfail))
    msg = f'{name}: {total} events in {nfiles} files ({nfail} failed to open)'
    if first_error:
        msg += f'  [first error: {first_error}]'
    print(msg, flush=True)

with open(f'processed_events_{mode}.txt', 'w') as f:
    for name, total, nfiles, nfail in results:
        f.write(f'{name} {total} {nfiles} {nfail}\n')

total_events = sum(r[1] for r in results)
total_files = sum(r[2] for r in results)
total_fail = sum(r[3] for r in results)
print(f'\nTotal across all: {total_events} events in {total_files} files ({total_fail} failed to open)')