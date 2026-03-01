#!/usr/bin/env python3
"""
Public dataset downloader/preparer for Hyundai Seg-NAS experiments.

Examples:
  # ADE20K auto download
  python hyundai/scripts/download_public_datasets.py \
      --dataset ade20k --output_dir ./dataset/public/ade20k

  # Cityscapes auto download (requires account + access rights)
  python hyundai/scripts/download_public_datasets.py \
      --dataset cityscapes --output_dir ./dataset/public/cityscapes

  # MVTec AD / LOCO / VisA auto download
  python hyundai/scripts/download_public_datasets.py \
      --dataset mvtec_ad --output_dir ./dataset/public/mvtec_ad

  # MVTec AD + LOCO + VisA in one command
  python hyundai/scripts/download_public_datasets.py \
      --dataset all_industrial --output_dir ./dataset/public
"""

import argparse
import os
import glob
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from http.cookiejar import CookieJar


AUTO_URLS = {
    "ade20k": [
        "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
    ],
    # Sources:
    # - MVTec AD / LOCO: MVTec official download pages (mydrive direct links)
    # - VisA: amazon-science/spot-diff README (AWS S3 link)
    "mvtec_ad": [
        "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz",
    ],
    "mvtec_loco": [
        "https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz",
    ],
    "visa": [
        "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
    ],
}


MANUAL_LINKS = {
    "cityscapes": "https://www.cityscapes-dataset.com/downloads/",
    "mvtec_ad": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
    "mvtec_loco": "https://www.mvtec.com/company/research/datasets/mvtec-loco",
    "visa": "https://github.com/amazon-science/spot-diff",
}

CITYSCAPES_LOGIN_URL = "https://www.cityscapes-dataset.com/login/"
CITYSCAPES_DOWNLOAD_URL = "https://www.cityscapes-dataset.com/file-handling/?packageID={package_id}"
CITYSCAPES_PACKAGE_IDS = {
    "gtFine_trainvaltest.zip": 1,
    "leftImg8bit_trainvaltest.zip": 3,
}
INDUSTRIAL_DATASETS = ["mvtec_ad", "mvtec_loco", "visa"]


def _is_visa_layout_ready(base_dir: Path) -> bool:
    # Official VisA archive usually contains split_csv and 1cls/2cls folders.
    return (
        (base_dir / "split_csv").exists()
        or (base_dir / "1cls").exists()
        or (base_dir / "2cls").exists()
    )


def _dataset_ready(dataset: str, output_dir: Path) -> bool:
    if dataset == "cityscapes":
        return (
            (output_dir / "leftImg8bit" / "train").exists()
            and (output_dir / "gtFine" / "train").exists()
        )
    if dataset == "ade20k":
        return (
            (output_dir / "ADEChallengeData2016" / "images" / "training").exists()
            and (output_dir / "ADEChallengeData2016" / "annotations" / "training").exists()
        )
    if dataset in {"mvtec_ad", "mvtec_loco"}:
        if not output_dir.exists():
            return False
        for p in output_dir.iterdir():
            if p.is_dir() and (p / "train").exists() and (p / "test").exists() and (p / "ground_truth").exists():
                return True
        return False
    if dataset == "visa":
        if not output_dir.exists():
            return False
        # Ignore hidden artifacts and downloader temp directory.
        visible_entries = [
            p for p in output_dir.iterdir() if p.name != "downloads" and not p.name.startswith(".")
        ]
        if not visible_entries:
            return False
        if _is_visa_layout_ready(output_dir):
            return True
        # Common case: extracted into nested root like VisA_20220922/.
        for p in visible_entries:
            if p.is_dir() and _is_visa_layout_ready(p):
                return True
        return False
    return False


def _filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    return name or "download.bin"


DATASET_URL_ENV_KEYS = {
    "ade20k": ["ADE20K_URL", "ADE20K_URLS"],
    "mvtec_ad": ["MVTEC_AD_URL", "MVTEC_AD_URLS"],
    "mvtec_loco": ["MVTEC_LOCO_URL", "MVTEC_LOCO_URLS"],
    "visa": ["VISA_URL", "VISA_URLS"],
}


def _parse_url_list(raw: str) -> list[str]:
    if not raw:
        return []
    for sep in [",", ";", "\n", "\t", " "]:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep)]
            return [p for p in parts if p]
    return [raw.strip()] if raw.strip() else []


def _resolve_auto_urls(dataset: str) -> list[str]:
    # Environment variable override first, then built-in defaults.
    for key in DATASET_URL_ENV_KEYS.get(dataset, []):
        val = os.getenv(key)
        parsed = _parse_url_list(val) if val else []
        if parsed:
            return parsed
    return list(AUTO_URLS.get(dataset, []))


def _is_html_file(path: Path) -> bool:
    try:
        head = path.read_bytes()[:2048].lower()
    except Exception:
        return False
    return (b"<!doctype html" in head) or (b"<html" in head)


def _is_valid_archive(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".zip"):
        return zipfile.is_zipfile(path)
    if any(name.endswith(ext) for ext in [".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2"]):
        return tarfile.is_tarfile(path)
    return False


def _ensure_archive_or_raise(path: Path):
    if _is_valid_archive(path):
        return
    if _is_html_file(path):
        raise RuntimeError(
            f"Downloaded file is HTML (not archive): {path}\n"
            "Cityscapes login/session likely failed or account permission is missing."
        )
    raise RuntimeError(f"Downloaded file is not a valid archive: {path}")


def _download(url: str, dst: Path, force: bool = False):
    if dst.exists() and not force:
        if _is_valid_archive(dst):
            print(f"Skipping download (exists): {dst}")
            return
        print(f"Existing file is invalid, re-downloading: {dst}")
        dst.unlink(missing_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response, open(dst, "wb") as f:
        shutil.copyfileobj(response, f)
    _ensure_archive_or_raise(dst)
    print(f"Saved: {dst}")


def _download_with_opener(opener, url: str, dst: Path, force: bool = False):
    if dst.exists() and not force:
        if _is_valid_archive(dst):
            print(f"Skipping download (exists): {dst}")
            return
        print(f"Existing file is invalid, re-downloading: {dst}")
        dst.unlink(missing_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with opener.open(req) as response, open(dst, "wb") as f:
            shutil.copyfileobj(response, f)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error while downloading {url}: {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error while downloading {url}: {e.reason}") from e
    _ensure_archive_or_raise(dst)


def _download_cityscapes_archives(
    output_dir: Path,
    username: str,
    password: str,
    package_names,
    force_download: bool = False,
):
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    login_payload = urllib.parse.urlencode(
        {
            "username": username,
            "password": password,
            "submit": "Login",
        }
    ).encode("utf-8")
    login_req = urllib.request.Request(
        CITYSCAPES_LOGIN_URL,
        data=login_payload,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with opener.open(login_req) as _:
            pass
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Cityscapes login failed: HTTP {e.code} {e.reason}. "
            "Check account credentials and access permissions."
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cityscapes login failed due to network error: {e.reason}") from e

    download_dir = output_dir / "downloads"
    downloaded = []
    for pkg_name in package_names:
        if pkg_name not in CITYSCAPES_PACKAGE_IDS:
            valid = ", ".join(sorted(CITYSCAPES_PACKAGE_IDS.keys()))
            raise ValueError(f"Unknown Cityscapes package '{pkg_name}'. Valid: {valid}")
        pkg_id = CITYSCAPES_PACKAGE_IDS[pkg_name]
        url = CITYSCAPES_DOWNLOAD_URL.format(package_id=pkg_id)
        dst = download_dir / pkg_name
        print(f"Downloading Cityscapes package: {pkg_name} (packageID={pkg_id})")
        _download_with_opener(opener, url, dst, force=force_download)
        downloaded.append(dst)

    return downloaded


def _extract(archive: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    _ensure_archive_or_raise(archive)
    print(f"Extracting: {archive}")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
        return

    if any(name.endswith(ext) for ext in [".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2"]):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(out_dir)
        return

    raise ValueError(f"Unsupported archive format: {archive}")


def _post_layout_hint(dataset: str, output_dir: Path):
    print("\nExpected data layout for this repository:")
    if dataset == "cityscapes":
        print(f"- {output_dir}/leftImg8bit/train|val/... and {output_dir}/gtFine/train|val/...")
    elif dataset == "ade20k":
        print(
            f"- {output_dir}/ADEChallengeData2016/images/training|validation and "
            f"{output_dir}/ADEChallengeData2016/annotations/training|validation"
        )
    elif dataset in {"mvtec_ad", "mvtec_loco"}:
        print(f"- {output_dir}/<category>/train|test|ground_truth")
    elif dataset == "visa":
        print(
            f"- Either MVTec-like folders under {output_dir}, "
            f"or CSV manifest + relative image/mask paths"
        )


def _looks_like_placeholder(path_str: str) -> bool:
    normalized = path_str.strip().lower()
    return normalized.startswith("/path/to/") or normalized.startswith("path/to/")


def _prepare_single_dataset(dataset: str, output_dir: Path, args, cli_urls=None, cli_archives=None):
    cli_urls = list(cli_urls or [])
    cli_archives = list(cli_archives or [])
    output_dir.mkdir(parents=True, exist_ok=True)

    if _dataset_ready(dataset, output_dir) and not args.force_extract and not cli_urls and not cli_archives:
        print(f"Dataset already prepared at: {output_dir}")
        _post_layout_hint(dataset, output_dir)
        return

    urls = list(cli_urls)
    if not urls:
        urls = _resolve_auto_urls(dataset)

    downloaded_archives = []
    if urls:
        download_dir = output_dir / "downloads"
        for url in urls:
            filename = _filename_from_url(url)
            dst = download_dir / filename
            _download(url, dst, force=args.force_download)
            downloaded_archives.append(dst)
    elif (
        dataset == "cityscapes"
        and not cli_archives
        and args.cityscapes_username
        and args.cityscapes_password
    ):
        downloaded_archives.extend(
            _download_cityscapes_archives(
                output_dir=output_dir,
                username=args.cityscapes_username,
                password=args.cityscapes_password,
                package_names=args.cityscapes_packages,
                force_download=args.force_download,
            )
        )
    elif dataset in MANUAL_LINKS and not cli_archives:
        print(f"No direct URL provided for '{dataset}'.")
        print(f"Please download archives manually from: {MANUAL_LINKS[dataset]}")
        if dataset == "cityscapes":
            print("Or provide login credentials for auto download:")
            print("  CITYSCAPES_USERNAME=... CITYSCAPES_PASSWORD=... python ... --dataset cityscapes ...")

    archive_paths = []
    missing_inputs = []
    for archive_arg in cli_archives:
        expanded = str(Path(archive_arg).expanduser())
        matches = sorted(glob.glob(expanded))
        if matches:
            archive_paths.extend([Path(m).resolve() for m in matches])
        else:
            missing_inputs.append(archive_arg)

    archive_paths.extend(downloaded_archives)

    if not archive_paths:
        if _dataset_ready(dataset, output_dir):
            print(f"Dataset already prepared at: {output_dir}")
            _post_layout_hint(dataset, output_dir)
            return

        if missing_inputs:
            details = "\n".join([f"  - {m}" for m in missing_inputs])
            msg = f"Archive path not found for:\n{details}\n"
            if any(_looks_like_placeholder(m) for m in missing_inputs):
                msg += (
                    "It looks like you used the example placeholder path.\n"
                    "Replace it with the real downloaded archive path(s).\n"
                    "Example:\n"
                    "  python hyundai/scripts/download_public_datasets.py \\\n"
                    "    --dataset cityscapes \\\n"
                    "    --output_dir ./dataset/public/cityscapes \\\n"
                    "    --archive ~/Downloads/leftImg8bit_trainvaltest.zip \\\n"
                    "    --archive ~/Downloads/gtFine_trainvaltest.zip\n"
                )
            raise FileNotFoundError(msg)

        print("No archives to extract. Stopping after guidance.")
        _post_layout_hint(dataset, output_dir)
        return

    if _dataset_ready(dataset, output_dir) and not args.force_extract:
        print(f"Dataset already prepared at: {output_dir}")
        print("Skipping extraction. Use --force_extract to re-extract archives.")
        if downloaded_archives and not args.keep_archives:
            shutil.rmtree(output_dir / "downloads", ignore_errors=True)
        _post_layout_hint(dataset, output_dir)
        return

    for archive_path in archive_paths:
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        _extract(archive_path, output_dir)

    if downloaded_archives and not args.keep_archives:
        shutil.rmtree(output_dir / "downloads", ignore_errors=True)

    _post_layout_hint(dataset, output_dir)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Download or unpack public segmentation datasets.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cityscapes", "ade20k", "mvtec_ad", "visa", "mvtec_loco", "all_industrial"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory where dataset is stored/extracted.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Direct URL to download archive(s). Can be repeated.",
    )
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        help="Local archive path to extract. Can be repeated.",
    )
    parser.add_argument(
        "--keep_archives",
        action="store_true",
        help="Keep downloaded archives in <output_dir>/downloads.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download archives even if they already exist in downloads/.",
    )
    parser.add_argument(
        "--force_extract",
        action="store_true",
        help="Re-extract archives even when dataset layout already exists.",
    )
    parser.add_argument(
        "--cityscapes_username",
        default=os.getenv("CITYSCAPES_USERNAME"),
        help="Cityscapes username (or set env CITYSCAPES_USERNAME).",
    )
    parser.add_argument(
        "--cityscapes_password",
        default=os.getenv("CITYSCAPES_PASSWORD"),
        help="Cityscapes password (or set env CITYSCAPES_PASSWORD).",
    )
    parser.add_argument(
        "--cityscapes_packages",
        nargs="+",
        default=["leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip"],
        help="Cityscapes package filenames to download (default: leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all_industrial":
        if args.url:
            raise ValueError(
                "--url is not supported with --dataset all_industrial. "
                "Use per-dataset env vars: MVTEC_AD_URL(S), MVTEC_LOCO_URL(S), VISA_URL(S)."
            )
        if args.archive:
            raise ValueError(
                "--archive is not supported with --dataset all_industrial. "
                "Run individual datasets if you need local archive paths."
            )
        print("Preparing industrial datasets: mvtec_ad, mvtec_loco, visa")
        for name in INDUSTRIAL_DATASETS:
            sub_output_dir = output_dir / name
            print(f"\n=== {name} ===")
            _prepare_single_dataset(
                dataset=name,
                output_dir=sub_output_dir,
                args=args,
                cli_urls=[],
                cli_archives=[],
            )
        print("\nAll industrial datasets done.")
        return

    _prepare_single_dataset(
        dataset=args.dataset,
        output_dir=output_dir,
        args=args,
        cli_urls=args.url,
        cli_archives=args.archive,
    )


if __name__ == "__main__":
    main()
