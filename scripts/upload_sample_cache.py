"""Mirror the local sample cache directory to GCS.

Walks every .pt.lz4 file under a local cache_dir and uploads to
gs://<bucket>/<prefix>/<filename>, skipping files that already exist on GCS
(by name; checks are HEAD requests in parallel). Sharded across replicas
via BEAKER_REPLICA_RANK / BEAKER_REPLICA_COUNT.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage
from tqdm import tqdm


def _parse_bucket(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI, got {uri}")
    rest = uri[len("gs://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="Local sample cache directory")
    ap.add_argument("--gcs_uri", required=True, help="Destination, e.g. gs://bucket/prefix")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--skip_existing", action="store_true", help="HEAD-check and skip already-uploaded objects")
    args = ap.parse_args()

    rank = int(os.environ.get("BEAKER_REPLICA_RANK", "0"))
    world = int(os.environ.get("BEAKER_REPLICA_COUNT", "1"))
    is_main = rank == 0

    bucket_name, prefix = _parse_bucket(args.gcs_uri)
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "no-project")
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    cache_dir = Path(args.cache_dir)
    if is_main:
        print(f"cache_dir: {cache_dir}")
        print(f"dest: gs://{bucket_name}/{prefix}/")
        print(f"replica: {rank}/{world}")
        print(f"workers: {args.workers}")
        print(f"skip_existing: {args.skip_existing}")

    # List local files, sharded by replica
    if is_main:
        print("scanning local files...")
    all_names = sorted(p.name for p in cache_dir.iterdir() if p.suffix == ".lz4")
    my_names = all_names[rank::world]
    if is_main:
        print(f"total local files: {len(all_names):,}")
    print(f"[rank {rank}] my share: {len(my_names):,}")

    # Optionally pre-list existing GCS objects under prefix so we can skip without HEAD-per-file
    existing: set[str] = set()
    if args.skip_existing:
        if is_main:
            print("listing existing GCS objects (this can take a few minutes)...")
        # All replicas list — duplicate work but cheap and avoids cross-replica coordination
        t0 = time.time()
        for blob in client.list_blobs(bucket_name, prefix=prefix + "/"):
            name = blob.name.split("/")[-1]
            if name.endswith(".pt.lz4"):
                existing.add(name)
        if is_main:
            print(f"existing GCS objects: {len(existing):,} (listed in {time.time()-t0:.1f}s)")

    todo = [n for n in my_names if n not in existing]
    print(f"[rank {rank}] todo (after skip): {len(todo):,}")

    def upload(name: str) -> tuple[str, bool, str]:
        try:
            blob = bucket.blob(f"{prefix}/{name}")
            blob.upload_from_filename(str(cache_dir / name))
            return name, True, ""
        except Exception as e:
            return name, False, str(e)

    ok = 0
    err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(upload, n) for n in todo]
        bar = tqdm(total=len(todo), desc=f"r{rank} upload", disable=not is_main, smoothing=0.05)
        for fut in as_completed(futs):
            _name, success, msg = fut.result()
            if success:
                ok += 1
            else:
                err += 1
                if err <= 10:
                    print(f"[rank {rank}] err {_name}: {msg}")
            bar.update(1)
        bar.close()

    dt = time.time() - t0
    rate = ok / dt if dt > 0 else 0
    print(f"[rank {rank}] done. ok={ok:,} err={err:,} elapsed={dt:.1f}s rate={rate:.1f}/s")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
