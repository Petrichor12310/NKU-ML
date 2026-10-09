"""Download pinned Weka runtime jars into this project (no system installation)."""
import hashlib
import json
from pathlib import Path
import urllib.request

BASE_URL = "https://repo.maven.apache.org/maven2/"
ARTIFACTS = [
    "nz/ac/waikato/cms/weka/weka-stable/3.8.6/weka-stable-3.8.6.jar",
    "nz/ac/waikato/cms/weka/thirdparty/bounce/0.18/bounce-0.18.jar",
    "com/github/vbmacher/java-cup-runtime/11b-20160615/java-cup-runtime-11b-20160615.jar",
]


def main():
    target = Path(__file__).resolve().parent / "tools/weka"
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for artifact in ARTIFACTS:
        url = BASE_URL + artifact
        expected = urllib.request.urlopen(url + ".sha1", timeout=60).read().decode().strip().split()[0]
        path = target / artifact.rsplit("/", 1)[-1]
        if not path.exists() or hashlib.sha1(path.read_bytes()).hexdigest() != expected:
            print(f"Downloading {url}", flush=True)
            data = urllib.request.urlopen(url, timeout=120).read()
            if hashlib.sha1(data).hexdigest() != expected:
                raise RuntimeError(f"Checksum mismatch for {url}")
            path.write_bytes(data)
        manifest.append({"file": path.name, "source": url,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (target / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Weka runtime ready: {target}")


if __name__ == "__main__":
    main()
