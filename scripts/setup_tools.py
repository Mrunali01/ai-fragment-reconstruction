"""Install local Windows development runtimes; no global PATH or services changed."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1] / ".tools"


def download(url: str, name: str) -> Path:
    path = ROOT / name
    if not path.exists():
        temporary = path.with_suffix(".download")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    version = "v24.22.0"
    name = f"node-{version}-win-x64.zip"
    base = f"https://nodejs.org/dist/{version}/"
    archive = download(base + name, name)
    checks = urllib.request.urlopen(base + "SHASUMS256.txt", timeout=30).read().decode()
    expected = next(line.split()[0] for line in checks.splitlines() if line.endswith("  " + name))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("Node archive checksum mismatch")
    with zipfile.ZipFile(archive) as package:
        package.extractall(ROOT)
    name = "postgresql-18.6-1-windows-x64-binaries.zip"
    archive = download("https://get.enterprisedb.com/postgresql/" + name, name)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            if member.filename.startswith(("pgsql/bin/", "pgsql/lib/", "pgsql/share/")):
                package.extract(member, ROOT)
    print("Local Node.js and PostgreSQL runtimes installed in .tools.")


if __name__ == "__main__":
    main()
