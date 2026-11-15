# %% [markdown]
# # Colab setup: run these cells at the start of EVERY Colab session
# Needs two Colab Secrets (key icon in the left sidebar, "Notebook access" ON):
#   OPENAQ_API_KEY  - from explore.openaq.org
#   GITHUB_TOKEN    - fine-grained GitHub token, this repo only, Contents: Read and write
# The repo lives in Google Drive, so data and results survive when Colab disconnects.

# %%  Cell 1 - settings (edit once)
GITHUB_USER = "mkhaliliiu"
REPO = "pm25-lahore"
GIT_NAME = "Muhammad Khalil"
GIT_EMAIL = "mkiiu47@gmail.com"          
DRIVE_DIR = "/content/drive/MyDrive"

# %%  Cell 2 - mount Drive, clone the repo the first time, enter it
import os, subprocess
from google.colab import drive, userdata

drive.mount("/content/drive")
REPO_DIR = f"{DRIVE_DIR}/{REPO}"
TOKEN = userdata.get("GITHUB_TOKEN")
URL = f"https://x-access-token:{TOKEN}@github.com/{GITHUB_USER}/{REPO}.git"

if not os.path.exists(REPO_DIR):
    r = subprocess.run(["git", "clone", "-q", URL, REPO_DIR], capture_output=True, text=True)
    print("clone ok" if r.returncode == 0 else r.stderr.replace(TOKEN, "***"))
    # keep the token out of .git/config on Drive
    subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin",
                    f"https://github.com/{GITHUB_USER}/{REPO}.git"])

os.chdir(REPO_DIR)
subprocess.run(["git", "config", "user.name", GIT_NAME])
subprocess.run(["git", "config", "user.email", GIT_EMAIL])
print("Working in:", os.getcwd())

# %%  Cell 3 - make the OpenAQ key available to scripts, install packages
os.environ["OPENAQ_API_KEY"] = userdata.get("OPENAQ_API_KEY")
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

# %%  Cell 4 - helper: pull and push without ever printing the token
def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(TOKEN, "***"))

def pull():
    git("pull", "-q", URL, "main")

def push(message):
    git("add", "-A")
    git("status", "--short")                 # check: .env must NOT appear here
    git("commit", "-q", "-m", message)
    git("push", "-q", URL, "HEAD:main")

pull()
