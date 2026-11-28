# === 2. Clone repo `Sper-il/COU` ===

import os
import subprocess

REPO_URL = "https://github.com/Sper-il/COU.git"
WORKDIR = "/content/COU"

if not os.path.isdir(WORKDIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, WORKDIR], check=True)
else:
    subprocess.run(["git", "-C", WORKDIR, "pull"], check=False)

os.chdir(WORKDIR)
print("thư mục hiện tại:", os.getcwd())
print(sorted(os.listdir("."))[:20])
