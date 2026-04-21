#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ftplib import FTP_TLS
from pathlib import Path

# ============================================================
# DIRECT LOGIN — PUT YOUR REAL TRACKMAN PASSWORD BELOW
# ============================================================

def connect_trackman():
    user = "Fordham"
    pwd  = "password"   # ← TYPE IT HERE

    ftp = FTP_TLS()
    ftp.connect("ftp.trackmanbaseball.com", 21)
    ftp.auth()
    ftp.prot_p()
    ftp.login(user, pwd)

    print(f"Connected to TrackMan FTP as {user}")
    return ftp


# ============================================================
# DOWNLOAD ALL YEARS / MONTHS / DAYS
# ============================================================

def download_years(ftp, years, base_dir):
    base_dir.mkdir(exist_ok=True)

    for year in years:
        print(f"\n=== YEAR {year} ===")
        for month in range(1, 13):
            for day in range(1, 32):

                remote_path = f"/v3/{year}/{month:02d}/{day:02d}/CSV"

                try:
                    ftp.cwd(remote_path)
                except Exception:
                    continue  # skip missing days

                local_day_dir = base_dir / year / f"{month:02d}" / f"{day:02d}"
                local_day_dir.mkdir(parents=True, exist_ok=True)

                try:
                    files = ftp.nlst()
                except Exception:
                    continue

                for filename in files:
                    if not filename.lower().endswith(".csv"):
                        continue

                    local_path = local_day_dir / filename

                    if local_path.exists():
                        print(f"Already exists, skipping: {local_path}")
                        continue

                    with open(local_path, "wb") as f:
                        ftp.retrbinary(f"RETR " + filename, f.write)

                    print(f"Downloaded: {local_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    base_dir = Path.home() / "Desktop" / "fordham_raw_data"
    years = ["2022-2026"]

    ftp = connect_trackman()

    try:
        download_years(ftp, years, base_dir)
    finally:
        ftp.quit()
        print("\nDisconnected from TrackMan FTP.")

    print("\nAll files downloaded successfully.")
    print(f"Saved under: {base_dir}")


if __name__ == "__main__":
    main()
