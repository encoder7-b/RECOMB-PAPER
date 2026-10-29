"""Pull iAtlas-harmonized ICI cohorts (TPM expression + clinical) from the public cBioPortal API.

Usage: python data/fetch_cbio.py [study_id ...]
Writes data/raw/<study_id>/{expr_tpm.csv.gz, clinical_sample.csv, clinical_patient.csv}.
"""
import json
import os
import sys
import urllib.request

import pandas as pd

API = "https://www.cbioportal.org/api"
RAW = os.path.join(os.path.dirname(__file__), "raw")
STUDIES = ["blca_iatlas_imvigor210_2017", "mel_iatlas_gide_2019", "mel_iatlas_liu_2019",
           "rcc_iatlas_immotion150_2018"]
CHUNK = 1000


def get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=300) as r:
        return json.load(r)


def post(path, body):
    req = urllib.request.Request(f"{API}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def entrez_ids():
    h = pd.read_csv(os.path.join(RAW, "hgnc", "hgnc_complete_set.txt"), sep="\t", dtype=str,
                    usecols=["symbol", "entrez_id", "locus_group"])
    h = h.dropna(subset=["entrez_id"])
    return dict(zip(h.entrez_id.astype(int), h.symbol))


def clinical(study, kind):
    rows = get(f"/studies/{study}/clinical-data?clinicalDataType={kind}&projection=SUMMARY")
    key = "sampleId" if kind == "SAMPLE" else "patientId"
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.pivot_table(index=[key] + (["patientId"] if kind == "SAMPLE" else []),
                          columns="clinicalAttributeId", values="value", aggfunc="first").reset_index()


def expression(study, genes):
    prof = f"{study}_rna_seq_mrna"
    ids = list(genes)
    parts = []
    for i in range(0, len(ids), CHUNK):
        rows = post(f"/molecular-profiles/{prof}/molecular-data/fetch?projection=SUMMARY",
                    {"sampleListId": f"{study}_all", "entrezGeneIds": ids[i:i + CHUNK]})
        if rows:
            parts.append(pd.DataFrame(rows)[["entrezGeneId", "sampleId", "value"]])
        print(f"  {study}: genes {min(i + CHUNK, len(ids))}/{len(ids)}", flush=True)
    long = pd.concat(parts)
    wide = long.pivot_table(index="entrezGeneId", columns="sampleId", values="value", aggfunc="first")
    wide.index = wide.index.map(genes)
    return wide


def main():
    genes = entrez_ids()
    for study in sys.argv[1:] or STUDIES:
        out = os.path.join(RAW, study)
        os.makedirs(out, exist_ok=True)
        clinical(study, "SAMPLE").to_csv(f"{out}/clinical_sample.csv", index=False)
        clinical(study, "PATIENT").to_csv(f"{out}/clinical_patient.csv", index=False)
        if not os.path.exists(f"{out}/expr_tpm.csv.gz"):
            x = expression(study, genes)
            x.to_csv(f"{out}/expr_tpm.csv.gz")
            print(f"{study}: expression {x.shape}", flush=True)


if __name__ == "__main__":
    main()
