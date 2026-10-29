"""Download public ICI cohorts and external reference expression into data/raw/.

Each source goes into its own empty directory. Nothing here is executed from
the downloaded content; parsing is done by data/prepare.py with paths passed in.

Usage:  python data/fetch.py --list         # show what would be fetched
        python data/fetch.py [names...]     # fetch (all if none given)
"""
import argparse
import os
import re
import sys
import urllib.request

RAW = os.path.join(os.path.dirname(__file__), "raw")

GEO = {
    # name: (GSE, regex over supplementary file names)
    "hugo2016": ("GSE78220", r".*"),
    "riaz2017": ("GSE91061", r".*"),
    "jung2019": ("GSE135222", r".*"),
    "gse176307": ("GSE176307", r".*"),
}

XENA = "https://tcga.xenahubs.net/download"
EXTERNAL = {  # TCGA reference expression (log2(RSEM+1)), matched tumor types
    f"tcga_{t.lower()}": f"{XENA}/TCGA.{t}.sampleMap/HiSeqV2.gz"
    for t in ("SKCM", "STAD", "LUAD", "LUSC", "BLCA", "KIRC")
}

DIRECT = {  # supplementary files fetched by URL
    "braun2020": ["https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-020-0839-y/"
                  "MediaObjects/41591_2020_839_MOESM2_ESM.xlsx"],
}

MANUAL = {
    "kim2018": "ENA PRJEB25780: raw FASTQ only (not used).",
    "gide/liu/imvigor210/immotion150": "fetched from cBioPortal iAtlas by data/fetch_cbio.py",
}


def geo_dir(gse):
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}"


def list_geo(gse, pat):
    html = urllib.request.urlopen(f"{geo_dir(gse)}/suppl/", timeout=60).read().decode()
    files = sorted(set(re.findall(r'href="([^"/?]+)"', html)))
    files = [f for f in files if re.fullmatch(pat, f)]
    return [f"{geo_dir(gse)}/suppl/{f}" for f in files] + [
        f"{geo_dir(gse)}/matrix/{gse}_series_matrix.txt.gz"]


def fetch(url, outdir):
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, url.rsplit("/", 1)[-1])
    if os.path.exists(dest):
        print("  have", dest)
        return
    print("  get ", url, flush=True)
    try:
        urllib.request.urlretrieve(url, dest + ".part")
        os.replace(dest + ".part", dest)
    except Exception as e:  # series matrix can be split into -GPLxxx files
        print("  FAILED", url, e, file=sys.stderr)


def plan():
    out = {}
    for name, (gse, pat) in GEO.items():
        out[name] = list_geo(gse, pat)
    for name, url in EXTERNAL.items():
        out[name] = [url]
    out.update(DIRECT)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    todo = plan()
    if args.names:
        todo = {k: v for k, v in todo.items() if k in args.names}
    for name, urls in todo.items():
        print(name)
        for u in urls:
            if args.list:
                print("  ", u)
            else:
                fetch(u, os.path.join(RAW, name))
    if args.list:
        print("\nManual sources (not fetched automatically):")
        for k, v in MANUAL.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
