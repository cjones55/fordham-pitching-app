#!/usr/bin/env python3
"""Download 2026 TrackMan game CSVs into scouting_2026_trackman.

This is meant for the large one-time import. It avoids Streamlit timeouts and can be rerun safely.
Passwords are prompted at runtime unless FTP_PASSWORD is set in the environment.
"""

import argparse
import ftplib
import getpass
import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scouting_2026_trackman"

EXCLUDED_TERMS = (
    "unverified", "practice", "playerpositioning", "positional", "position",
    "bullpen", "scrimmage", "intrasquad", "test"
)


def should_import(path: str) -> bool:
    name = Path(path).name.lower()
    full = path.lower()
    return bool(re.match(r"^2026\d{4}-.+-\d+\.csv$", name)) and not any(term in full for term in EXCLUDED_TERMS)


def validate_csv(path: Path) -> bool:
    try:
        sample = pd.read_csv(path, nrows=25, encoding="latin1", sep=None, engine="python")
    except Exception:
        return False
    return {"Pitcher", "Batter", "PitchCall", "TaggedPitchType"}.issubset(sample.columns)


def join_remote(parent: str, child: str) -> str:
    parent = parent.rstrip("/")
    if not parent:
        return f"/{child.strip('/')}"
    return f"{parent}/{child.strip('/')}"


def roots(base: str, months, day, csv_folder):
    base = base.strip() or "/"
    if not base.startswith("/"):
        base = f"/{base}"
    base = base.rstrip("/")
    csv_folder = (csv_folder or "CSV").strip("/")
    if not months:
        months = [f"{m:02d}" for m in range(1, 13)]
    for month in months:
        month_root = join_remote(base, str(month).zfill(2))
        days = [str(day).zfill(2)] if day else [f"{d:02d}" for d in range(1, 32)]
        for d in days:
            yield join_remote(join_remote(month_root, d), csv_folder)


def safe_name(remote_path: str) -> str:
    return remote_path.strip("/").replace("/", "__").replace(" ", "_")


def ftp_connect(args, password):
    cls = ftplib.FTP_TLS if args.protocol == "ftps" else ftplib.FTP
    ftp = cls()
    ftp.connect(args.host, args.port, timeout=args.timeout)
    ftp.login(args.username, password)
    ftp.set_pasv(args.passive)
    if args.protocol == "ftps":
        ftp.prot_p()
    return ftp


def sftp_connect(args, password):
    try:
        import paramiko
    except ImportError:
        print("SFTP requires paramiko. Install with: pip install paramiko", file=sys.stderr)
        raise
    transport = paramiko.Transport((args.host, args.port))
    transport.banner_timeout = args.timeout
    transport.auth_timeout = args.timeout
    transport.connect(username=args.username, password=password)
    return transport, paramiko.SFTPClient.from_transport(transport)


def download_ftp(args, password):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imported = skipped = scanned = 0
    ftp = ftp_connect(args, password)
    try:
        for root in roots(args.remote_dir, args.month, args.day, args.csv_folder):
            try:
                names = ftp.nlst(root)
            except Exception as exc:
                print(f"skip folder {root}: {exc}")
                continue
            for item in names:
                remote = item if str(item).startswith("/") else join_remote(root, item)
                scanned += 1
                if not should_import(remote):
                    skipped += 1
                    continue
                target = OUT_DIR / safe_name(remote)
                if args.skip_existing and target.exists():
                    skipped += 1
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp_path = Path(tmp.name)
                    try:
                        ftp.retrbinary(f"RETR {remote}", tmp.write)
                    except Exception as exc:
                        print(f"download failed {remote}: {exc}")
                        skipped += 1
                        tmp_path.unlink(missing_ok=True)
                        continue
                if not validate_csv(tmp_path):
                    skipped += 1
                    tmp_path.unlink(missing_ok=True)
                    continue
                target.write_bytes(tmp_path.read_bytes())
                tmp_path.unlink(missing_ok=True)
                imported += 1
                if imported % 100 == 0:
                    print(f"imported {imported} files... scanned {scanned}")
                if args.max_downloads and imported >= args.max_downloads:
                    return imported, skipped, scanned
    finally:
        ftp.quit()
    return imported, skipped, scanned


def download_sftp(args, password):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imported = skipped = scanned = 0
    transport, sftp = sftp_connect(args, password)
    try:
        for root in roots(args.remote_dir, args.month, args.day, args.csv_folder):
            try:
                entries = sftp.listdir(root)
            except Exception as exc:
                print(f"skip folder {root}: {exc}")
                continue
            for name in entries:
                remote = join_remote(root, name)
                scanned += 1
                if not should_import(remote):
                    skipped += 1
                    continue
                target = OUT_DIR / safe_name(remote)
                if args.skip_existing and target.exists():
                    skipped += 1
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    sftp.get(remote, str(tmp_path))
                except Exception as exc:
                    print(f"download failed {remote}: {exc}")
                    skipped += 1
                    tmp_path.unlink(missing_ok=True)
                    continue
                if not validate_csv(tmp_path):
                    skipped += 1
                    tmp_path.unlink(missing_ok=True)
                    continue
                target.write_bytes(tmp_path.read_bytes())
                tmp_path.unlink(missing_ok=True)
                imported += 1
                if imported % 100 == 0:
                    print(f"imported {imported} files... scanned {scanned}")
                if args.max_downloads and imported >= args.max_downloads:
                    return imported, skipped, scanned
    finally:
        sftp.close()
        transport.close()
    return imported, skipped, scanned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["ftp", "ftps", "sftp"], default="ftp")
    parser.add_argument("--host", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--port", type=int, default=21)
    parser.add_argument("--remote-dir", default="v3/2026")
    parser.add_argument("--csv-folder", default="CSV")
    parser.add_argument("--month", action="append", default=[])
    parser.add_argument("--day", default="")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-downloads", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--passive", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    password = os.environ.get("FTP_PASSWORD") or getpass.getpass("Password: ")
    if args.protocol == "sftp" and args.port == 21:
        print("Note: SFTP usually uses port 22. For TrackMan FileZilla FTP, use --protocol ftp --port 21.")
    if args.protocol == "sftp":
        imported, skipped, scanned = download_sftp(args, password)
    else:
        imported, skipped, scanned = download_ftp(args, password)
    print(f"Done. scanned={scanned} imported={imported} skipped={skipped} output={OUT_DIR}")


if __name__ == "__main__":
    main()
