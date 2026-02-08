#!/usr/bin/env python3
########################################################################
# Copyright (C) 2026  Kevin M. Hubbard BlackMesaLabs
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This is part of SUMP3 project: https://github.com/blackmesalabs/sump3
# 2026.02.07 : Created
########################################################################
import os
import hmac
import hashlib
import base64
import sys

def load_site_secret_key(ash_dir):
    """Load the site secret key from $ASH_DIR/site_key.txt."""
    key_path = os.path.join(ash_dir, "site_key.txt")
    try:
        with open(key_path, "r") as f:
            key = f.read().strip()
            if not key:
                print("Error: site_key.txt is empty.")
                sys.exit(1)
#           return key.encode("utf-8")
            return key
    except FileNotFoundError:
        print(f"Error: site_key.txt not found in {ash_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading site_key.txt: {e}")
        sys.exit(1)


def xor_bytes(data, key):
    key = key * (len(data) // len(key) + 1)
    return bytes([a ^ b for a, b in zip(data, key)])

def generate_encrypted_api_key(secret_key, username, api_key):
    key_bytes = secret_key.encode()
    api_bytes = api_key.encode()

    cipher = xor_bytes(api_bytes, key_bytes)
    cipher_hex = cipher.hex()

    payload = f"{username}|{cipher_hex}"
    sig = hmac.new(key_bytes, payload.encode(), hashlib.sha256).hexdigest()

    return f"{payload}|{sig}"

def main():
    if len(sys.argv) != 3:
        print("Usage: ash_token_maker.py <username> <REAL_API_KEY>")
        sys.exit(1)

    username = sys.argv[1]
    real_api_key = sys.argv[2]

    ash_dir = os.environ.get("ASH_DIR")
    if not ash_dir:
        print("Error: ASH_DIR environment variable is not set.")
        sys.exit(1)

    site_secret_key = load_site_secret_key(ash_dir)
    blob = generate_encrypted_api_key(site_secret_key, username, real_api_key )

    print("\nEncrypted API key blob for user:", username)
    print("--------------------------------------------")
    print(blob)
    print("--------------------------------------------")
    print("Place this blob into the user's ASH_TOKEN environment variable.\n")


if __name__ == "__main__":
    main()
