import sys, os, subprocess, pathlib
rar_path = sys.argv[1]
out_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/raw")
out_dir.mkdir(parents=True, exist_ok=True)
# Requires 7-Zip (p7zip-full on Linux, 7-Zip on Windows, brew install p7zip on macOS)
subprocess.check_call(["7z", "x", rar_path, f"-o{out_dir}", "-y"])
print(f"Extracted to {out_dir}")
