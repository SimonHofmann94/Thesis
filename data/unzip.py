#!/usr/bin/env python3
import sys
import os
from zipfile import ZipFile

def usage():
    print(f"Usage: {sys.argv[0]} <archive.zip> [<destination_folder>]")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()

    zip_path = sys.argv[1]
    if not os.path.isfile(zip_path):
        print(f"Error: '{zip_path}' not found.")
        sys.exit(1)

    # Zielordner: entweder das 2. Argument oder der Name der ZIP (ohne .zip)
    dest = sys.argv[2] if len(sys.argv) >= 3 else os.path.splitext(zip_path)[0]

    # Ordner anlegen, falls nicht vorhanden
    os.makedirs(dest, exist_ok=True)

    # Entpacken
    with ZipFile(zip_path, 'r') as archive:
        archive.extractall(dest)

    print(f"Erfolgreich entpackt '{zip_path}' nach '{dest}'")

if __name__ == "__main__":
    main()
