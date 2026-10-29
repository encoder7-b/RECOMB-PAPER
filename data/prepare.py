"""Label-blind preparation of each cohort into data/processed/.

Pre-specified rules (fixed before any modelling):
  expression : log2(FPKM/TPM + 1) gene symbols; TCGA kept as log2(RSEM + 1)
  samples    : pre-treatment only, one sample per patient (first listed)
  response   : 1 = CR/PR, 0 = SD/PD (NE/missing excluded)
               Jung (no RECIST in GEO): durable clinical benefit, 1 = PFS >= 180 d,
               0 = progression before 180 d, censored before 180 d excluded
  batch      : cohort-internal sub-batch where recorded (GSE176307: IO line 1 vs later)
Labels are written to a separate file from expression so sealed cohorts can be hashed.
"""
import gzip
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, OUT = os.path.join(HERE, "raw"), os.path.join(HERE, "processed")


def series_matrix(path):
    """Sample annotations from a GEO series matrix as a DataFrame (one row per GSM)."""
    rows, chars = {}, []
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            key, *vals = line.rstrip("\n").split("\t")
            vals = [v.strip('"') for v in vals]
            if key == "!Sample_characteristics_ch1":
                chars.append(vals)
            else:
                rows.setdefault(key[8:], vals)
    df = pd.DataFrame(rows)
    for vals in chars:  # "key: value" cells; keys can shift between samples
        for i, v in enumerate(vals):
            if ": " in v:
                k, val = v.split(": ", 1)
                df.loc[i, k.strip()] = val.strip()
    return df


def collapse_symbols(expr):
    """Sum duplicate symbols (transcript-level rows) and drop empties."""
    expr = expr[expr.index.notna() & (expr.index != "")]
    return expr.groupby(level=0).sum()


def recist(v):
    v = str(v).upper()
    if v in ("CR", "PR", "COMPLETE RESPONSE", "PARTIAL RESPONSE", "PRCR"):
        return 1.0
    if v in ("SD", "PD", "PROGRESSIVE DISEASE", "STABLE DISEASE"):
        return 0.0
    return np.nan


def hugo():
    meta = series_matrix(f"{RAW}/hugo2016/GSE78220_series_matrix.txt.gz")
    meta = meta[meta["biopsy time"] == "pre-treatment"]
    expr = pd.read_excel(f"{RAW}/hugo2016/GSE78220_PatientFPKM.xlsx").set_index("Gene")
    expr.columns = [c.split(".")[0] for c in expr.columns]
    lab = pd.DataFrame({"sample": meta["title"], "patient": meta["patient id"].str.replace(r"[ab]$", "", regex=True),
                        "y": meta["anti-pd-1 response"].map(recist), "batch": 0})
    return collapse_symbols(np.log2(expr + 1)), lab


def gse176307():
    meta = series_matrix(f"{RAW}/gse176307/GSE176307_series_matrix.txt.gz")
    meta["sample"] = meta["title"].str.replace("Patient sample ", "", regex=False)
    key = pd.read_csv(f"{RAW}/gse176307/GSE176307_BACI_Omniseq_Sample_Name_Key_submitted_GEO_v2.csv.gz")
    rs2baci = dict(zip(key.iloc[:, 1], key.iloc[:, 0]))
    expr = pd.read_csv(f"{RAW}/gse176307/GSE176307_salmon_tpm_gene.matrix.tsv.gz", sep="\t", index_col=0)
    expr.columns = [rs2baci.get(c, c) for c in expr.columns]
    lab = pd.DataFrame({"sample": meta["sample"], "patient": meta["sample"],
                        "y": meta["io.response"].map(recist),
                        "batch": (pd.to_numeric(meta["io.lot"], errors="coerce") > 1).astype(int)})
    return collapse_symbols(np.log2(expr + 1)), lab


def riaz(gene_map):
    meta = series_matrix(f"{RAW}/riaz2017/GSE91061_series_matrix.txt.gz")
    meta = meta[meta["visit (pre or on treatment)"] == "Pre"]
    expr = pd.read_csv(f"{RAW}/riaz2017/GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz", index_col=0)
    expr.index = expr.index.astype(str).map(gene_map["entrez"])
    lab = pd.DataFrame({"sample": meta["title"], "patient": meta["title"].str.split("_").str[0],
                        "y": meta["response"].map(recist), "batch": 0})
    return collapse_symbols(np.log2(expr + 1)), lab


def jung(gene_map):
    meta = series_matrix(f"{RAW}/jung2019/GSE135222_series_matrix.txt.gz")
    meta["sample"] = meta["title"].str.replace(" ", "", regex=False)
    t = pd.to_numeric(meta["pfs.time"], errors="coerce")
    ev = pd.to_numeric(meta["progression-free survival (pfs)"], errors="coerce")
    y = np.where(t >= 180, 1.0, np.where(ev == 1, 0.0, np.nan))
    expr = pd.read_csv(f"{RAW}/jung2019/GSE135222_GEO_RNA-seq_omicslab_exp.tsv.gz", sep="\t", index_col=0)
    expr.index = expr.index.str.split(".").str[0].map(gene_map["ensembl"])
    lab = pd.DataFrame({"sample": meta["sample"], "patient": meta["sample"], "y": y, "batch": 0})
    return collapse_symbols(np.log2(expr + 1)), lab


def iatlas(study, select=None, batch=None):
    """iAtlas-harmonized cBioPortal study: TPM expression + RESPONSE (RECIST)."""
    d = f"{RAW}/{study}"
    smp = pd.read_csv(f"{d}/clinical_sample.csv", dtype={"sampleId": str, "patientId": str})
    pat = pd.read_csv(f"{d}/clinical_patient.csv", dtype={"patientId": str})
    meta = smp.merge(pat, on="patientId", how="left", suffixes=("", "_pt"))
    if "SAMPLE_TREATMENT" in meta:
        meta = meta[meta["SAMPLE_TREATMENT"].fillna("Pre") == "Pre"]
    if select is not None:
        meta = meta[select(meta)]
    expr = pd.read_csv(f"{d}/expr_tpm.csv.gz", index_col=0)
    resp = meta["RESPONSE"].map({"Complete Response": 1.0, "Partial Response": 1.0,
                                 "Stable Disease": 0.0, "Progressive Disease": 0.0})
    b = batch(meta) if batch is not None else 0
    lab = pd.DataFrame({"sample": meta["sampleId"], "patient": meta["patientId"], "y": resp, "batch": b})
    return collapse_symbols(np.log2(expr.fillna(0) + 1)), lab


def braun_extract():
    """Convert the Braun 2020 supplementary workbook (S1 clinical, S4A RNA) to CSV once."""
    d = f"{RAW}/braun2020"
    if os.path.exists(f"{d}/clinical.csv") and os.path.exists(f"{d}/rna_expression.csv.gz"):
        return
    xl = f"{d}/41591_2020_839_MOESM2_ESM.xlsx"
    pd.read_excel(xl, sheet_name="S1_Clinical_and_Immune_Data", header=1).to_csv(f"{d}/clinical.csv", index=False)
    pd.read_excel(xl, sheet_name="S4A_RNA_Expression", header=1, index_col=0).to_csv(f"{d}/rna_expression.csv.gz")


def braun(arm):
    """Braun 2020 (CheckMate 009/010/025). Expression as released by the authors (already
    normalized, log-scale); batch = trial. ORR: CR/PR/CRPR = 1, SD/PD = 0, NE excluded."""
    braun_extract()
    c = pd.read_csv(f"{RAW}/braun2020/clinical.csv", dtype=str)
    c = c[(c["Arm"] == arm) & (c["RNA_ID"].fillna("NA") != "NA")]
    expr = pd.read_csv(f"{RAW}/braun2020/rna_expression.csv.gz", index_col=0)
    y = c["ORR"].map({"CR": 1.0, "PR": 1.0, "CRPR": 1.0, "SD": 0.0, "PD": 0.0})
    lab = pd.DataFrame({"sample": c["RNA_ID"], "patient": c["SUBJID"], "y": y,
                        "batch": c["Cohort"].map({"CM-009": 0, "CM-010": 1, "CM-025": 2})})
    return collapse_symbols(expr), lab


IATLAS = {
    "imvigor210": dict(study="blca_iatlas_imvigor210_2017",
                       batch=lambda m: m["BIOPSY_SITE"].map({"Bladder": 0}).fillna(1).astype(int)
                       + m["BIOPSY_SITE"].isna().astype(int)),  # 0 bladder, 1 other site, 2 unknown
    "gide_pd1": dict(study="mel_iatlas_gide_2019", select=lambda m: m["ICI_PATHWAY"] == "PD1"),
    "gide_combo": dict(study="mel_iatlas_gide_2019", select=lambda m: m["ICI_PATHWAY"] == "CTLA4 PD1"),
    "liu2019": dict(study="mel_iatlas_liu_2019",
                    batch=lambda m: m["PRIOR_RX"].fillna("").str.contains("aCTLA4").astype(int)),
    "immotion_atezo": dict(study="rcc_iatlas_immotion150_2018",
                           select=lambda m: m["NON_ICI_RX"].fillna("none") != "Sunitinib",
                           batch=lambda m: (m["NON_ICI_RX"] == "Bevacizumab").astype(int)),
    "immotion_sunitinib": dict(study="rcc_iatlas_immotion150_2018",
                               select=lambda m: m["NON_ICI_RX"] == "Sunitinib"),
}


def load_gene_map():
    """HGNC complete set -> {entrez: symbol, ensembl: symbol}; None if not downloaded."""
    path = f"{RAW}/hgnc/hgnc_complete_set.txt"
    if not os.path.exists(path):
        return None
    h = pd.read_csv(path, sep="\t", dtype=str, usecols=["symbol", "entrez_id", "ensembl_gene_id"])
    return {"entrez": dict(zip(h.entrez_id, h.symbol)), "ensembl": dict(zip(h.ensembl_gene_id, h.symbol))}


def finish(name, expr, lab):
    lab = lab.dropna(subset=["y"])
    lab = lab[lab["sample"].isin(expr.columns)].drop_duplicates("patient", keep="first")
    expr = expr[lab["sample"]]
    os.makedirs(OUT, exist_ok=True)
    expr.to_csv(f"{OUT}/{name}_expr.csv.gz")
    lab.to_csv(f"{OUT}/{name}_labels.csv", index=False)
    print(f"{name:10s} n={len(lab):3d} responders={int(lab.y.sum()):3d} genes={expr.shape[0]} "
          f"batches={lab.batch.nunique()}")


def tcga():
    for t in ("skcm", "stad", "luad", "lusc", "blca", "kirc"):
        x = pd.read_csv(f"{RAW}/tcga_{t}/HiSeqV2.gz", sep="\t", index_col=0)
        x = x[[c for c in x.columns if c[13:15] in ("01", "06")]]  # tumour samples only
        os.makedirs(OUT, exist_ok=True)
        x.to_csv(f"{OUT}/tcga_{t}_expr.csv.gz")
        print(f"tcga_{t:5s} n={x.shape[1]} genes={x.shape[0]}")


def main():
    names = sys.argv[1:] or ["hugo", "gse176307", "riaz", "jung", *IATLAS, "braun_nivo", "braun_evero", "tcga"]
    gmap = load_gene_map()
    for nm in names:
        if nm == "tcga":
            tcga()
        elif nm in IATLAS:
            finish(nm, *iatlas(**IATLAS[nm]))
        elif nm == "braun_nivo":
            finish(nm, *braun("NIVOLUMAB"))
        elif nm == "braun_evero":
            finish(nm, *braun("EVEROLIMUS"))
        elif nm in ("riaz", "jung"):
            if gmap is None:
                print(f"{nm}: skipped, needs data/raw/hgnc/hgnc_complete_set.txt for ID mapping")
                continue
            finish(nm, *globals()[nm](gmap))
        else:
            finish(nm, *globals()[nm]())


if __name__ == "__main__":
    main()
