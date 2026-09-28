"""Download and validate the canonical AI4I 2020 Predictive Maintenance dataset from UCI."""

import io
import os
import urllib.request
import zipfile
import csv

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
TARGET_FILE = os.path.join(TARGET_DIR, "ai4i2020.csv")

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def download_and_validate():
    os.makedirs(TARGET_DIR, exist_ok=True)

    if os.path.exists(TARGET_FILE):
        print(f"Dataset already exists at: {TARGET_FILE}")
    else:
        print(f"Downloading canonical dataset from UCI: {UCI_ZIP_URL} ...")
        headers = {"User-Agent": "MachineGuard-AI-Downloader/1.0"}
        req = urllib.request.Request(UCI_ZIP_URL, headers=headers)
        with urllib.request.urlopen(req) as resp:
            content = resp.read()

        print(f"Downloaded {len(content)} bytes. Extracting ai4i2020.csv ...")
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            csv_candidates = [name for name in z.namelist() if name.endswith(".csv")]
            if not csv_candidates:
                raise RuntimeError("No CSV file found in UCI ZIP archive.")
            target_name = "ai4i2020.csv" if "ai4i2020.csv" in csv_candidates else csv_candidates[0]
            with z.open(target_name) as source, open(TARGET_FILE, "wb") as dest:
                dest.write(source.read())
        print(f"Saved dataset to {TARGET_FILE}")

    # Validate dataset content (use utf-8-sig to strip potential UTF-8 BOM)
    with open(TARGET_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Validating dataset...")
    print(f"Total records: {len(rows)} (Expected: 10,000)")
    assert len(rows) == 10000, f"Expected 10,000 records, got {len(rows)}"

    print(f"Columns found: {fieldnames}")
    for col in EXPECTED_COLUMNS:
        assert col in fieldnames, f"Missing required column: {col}"

    failures = sum(int(r["Machine failure"]) for r in rows)
    non_failures = len(rows) - failures
    print(f"Target breakdown: Failures={failures} ({failures/len(rows):.2%}), Non-failures={non_failures} ({non_failures/len(rows):.2%})")
    assert failures == 339, f"Expected exactly 339 failures, found {failures}"
    assert non_failures == 9661, f"Expected exactly 9661 non-failures, found {non_failures}"

    print("Canonical dataset validation PASSED successfully!")
    return TARGET_FILE


if __name__ == "__main__":
    download_and_validate()
