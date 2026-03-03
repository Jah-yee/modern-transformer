#!/usr/bin/env python3
"""Download Tiny Shakespeare dataset to data/tiny_shakespeare.txt."""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tiny_shakespeare.txt")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    urllib.request.urlretrieve(URL, OUTPUT_FILE)
    print(f"Downloaded to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
