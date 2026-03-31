#!/usr/bin/env python3
"""
Upload EfficientNet.py to GitHub using GitHub API
"""

import os
import base64
import requests
import json

# Configuration
REPO_OWNER = "Jung-woojin"
REPO_NAME = "CNN-From-Scratch-With-PyTorch"
FILE_PATH = "EfficientNet.py"
BRANCH = "main"
COMMIT_MESSAGE = "Implement EfficientNet-B0 from scratch with MBConv, SE blocks, and compound scaling"

# Read GitHub token from environment
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    print("Error: GITHUB_TOKEN environment variable is not set.")
    print("Please set your GitHub token and run again.")
    print("\nTo create a token:")
    print("1. Go to https://github.com/settings/tokens")
    print("2. Create a new token with 'repo' scope")
    print("3. Export it: export GITHUB_TOKEN=your_token_here")
    exit(1)

# Read the file
file_content = open(FILE_PATH, "r").read()

print(f"✓ Read file: {FILE_PATH}")
print(f"  Size: {len(file_content):,} bytes")
print(f"  Lines: {file_content.count(chr(10)) + 1}")

# Calculate SHA (git hash object)
def calculate_sha(content):
    """Calculate SHA-1 hash like Git does"""
    import hashlib
    sha1 = hashlib.sha1()
    sha1.update(f"blob {len(content)}\0".encode())
    sha1.update(content.encode("utf-8"))
    return sha1.hexdigest()

current_sha = calculate_sha(file_content)

# Get current file info
api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

print(f"\n🔍 Checking current file...")
print(f"  Repository: {REPO_OWNER}/{REPO_NAME}")
print(f"  Branch: {BRANCH}")

response = requests.get(api_url, headers=headers)

if response.status_code == 200:
    current_file = response.json()
    current_sha_remote = current_file.get("sha")
    
    print(f"  Current SHA: {current_sha_remote}")
    print(f"  Current Size: {current_file.get('size', 0):,} bytes")
    
    if current_sha == current_sha_remote:
        print("\n⚠️  File already up to date!")
        exit(0)
    else:
        print("\n📝 File will be updated")
        
elif response.status_code == 404:
    print("  File does not exist, will create new file")
    current_sha_remote = None
else:
    print(f"Error: {response.status_code} - {response.text}")
    exit(1)

# Prepare upload payload
payload = {
    "message": COMMIT_MESSAGE,
    "content": base64.b64encode(file_content.encode("utf-8")).decode("utf-8"),
    "branch": BRANCH,
    "sha": current_sha_remote
}

# Upload to GitHub
print(f"\n📤 Uploading to GitHub...")
response = requests.put(api_url, headers=headers, json=payload)

if response.status_code == 200:
    result = response.json()
    commit_sha = result.get("commit", {}).get("sha", "")
    commit_url = result.get("commit", {}).get("html_url", "")
    
    print("\n✅ Upload successful!")
    print(f"  Commit SHA: {commit_sha[:8]}")
    print(f"  Commit URL: {commit_url}")
    print(f"  Message: {COMMIT_MESSAGE}")
else:
    print(f"Error: {response.status_code} - {response.text}")
    exit(1)
