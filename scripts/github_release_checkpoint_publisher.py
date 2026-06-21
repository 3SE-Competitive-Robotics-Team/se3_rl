"""把本地 checkpoint 目录发布到 GitHub Release，供本机 Viser 稳定拉取。"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


@dataclass(frozen=True)
class LocalCheckpoint:
    path: Path
    iteration: int
    size: int
    mtime_ns: int


def github_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return fields.get("password") or None


def github_request(
    args: argparse.Namespace,
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    accept: str = "application/vnd.github+json",
    content_type: str | None = None,
) -> bytes:
    headers = {
        "Accept": accept,
        "User-Agent": "se3-checkpoint-publisher",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type is not None:
        headers["Content-Type"] = content_type
    if data is not None:
        headers["Content-Length"] = str(len(data))
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=float(args.github_timeout_s)) as response:
        return response.read()


def github_json(args: argparse.Namespace, url: str) -> object:
    return json.loads(github_request(args, url).decode("utf-8"))


def release(args: argparse.Namespace) -> dict[str, object]:
    url = (
        "https://api.github.com/repos/"
        f"{args.github_release_repo}/releases/tags/{args.github_release_tag}"
    )
    payload = github_json(args, url)
    if not isinstance(payload, dict):
        raise RuntimeError("unexpected GitHub release response")
    return payload


def latest_local_checkpoint(args: argparse.Namespace) -> LocalCheckpoint | None:
    checkpoints: list[LocalCheckpoint] = []
    if not args.checkpoint_dir.exists():
        return None
    for path in args.checkpoint_dir.glob("model_*.pt"):
        match = MODEL_RE.match(path.name)
        if match is None or not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size <= 0:
            continue
        checkpoints.append(
            LocalCheckpoint(
                path=path,
                iteration=int(match.group(1)),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item.iteration)


def stable_latest_checkpoint(args: argparse.Namespace) -> LocalCheckpoint | None:
    first = latest_local_checkpoint(args)
    if first is None or args.stability_seconds <= 0:
        return first
    time.sleep(args.stability_seconds)
    second = latest_local_checkpoint(args)
    if second is None:
        return None
    if (
        first.path == second.path
        and first.size == second.size
        and first.mtime_ns == second.mtime_ns
    ):
        return second
    return None


def release_asset_by_name(rel: dict[str, object], name: str) -> dict[str, object] | None:
    assets = rel.get("assets", [])
    if not isinstance(assets, list):
        return None
    for asset in assets:
        if isinstance(asset, dict) and asset.get("name") == name:
            return asset
    return None


def upload_checkpoint(args: argparse.Namespace, ckpt: LocalCheckpoint) -> None:
    rel = release(args)
    existing = release_asset_by_name(rel, ckpt.path.name)
    if existing is not None and int(existing.get("size", -1) or -1) == ckpt.size:
        print(f"[github-publisher] already uploaded {ckpt.path.name} size={ckpt.size}")
        return
    if existing is not None:
        asset_url = str(existing.get("url", ""))
        if asset_url:
            github_request(args, asset_url, method="DELETE")
            print(f"[github-publisher] deleted stale asset {ckpt.path.name}")
    upload_url = str(rel.get("upload_url", "")).split("{", 1)[0]
    if not upload_url:
        raise RuntimeError("GitHub release has no upload_url")
    query = urllib.parse.urlencode({"name": ckpt.path.name})
    data = ckpt.path.read_bytes()
    content_type = mimetypes.guess_type(ckpt.path.name)[0] or "application/octet-stream"
    payload = github_request(
        args,
        f"{upload_url}?{query}",
        method="POST",
        data=data,
        accept="application/vnd.github+json",
        content_type=content_type,
    )
    uploaded = json.loads(payload.decode("utf-8"))
    print(
        "[github-publisher] uploaded "
        f"{uploaded.get('name')} size={uploaded.get('size')} state={uploaded.get('state')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--github-release-repo",
        default="3SE-Competitive-Robotics-Team/se3_checkpoint_exchange",
    )
    parser.add_argument(
        "--github-release-tag",
        default="run-20260617-101355-stair3k-ctbcslow-f4ebc01",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stability-seconds", type=float, default=10.0)
    parser.add_argument("--interval-iters", type=int, default=100)
    parser.add_argument("--github-timeout-s", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    last_uploaded_iter = -1
    while True:
        ckpt = stable_latest_checkpoint(args)
        if ckpt is None:
            print("[github-publisher] no stable checkpoint yet")
        elif last_uploaded_iter < 0 or ckpt.iteration - last_uploaded_iter >= args.interval_iters:
            upload_checkpoint(args, ckpt)
            last_uploaded_iter = ckpt.iteration
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
