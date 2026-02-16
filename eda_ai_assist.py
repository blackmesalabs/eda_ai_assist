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
# The technical name is eda_ai_assist but the ChatBot is known as Ash.
#
# Env Setup Linux
#   set ash_path = ~/blah/python/cpy
#   alias ash '$ash_path/eda_ai_assist.py'
#  ~/.bashrc
#    export GEMINI_API_KEY=12345678
# Env Setup DOS
#  [ash.bat]
#  @echo off
#  C:\python310\python.exe "C:\foo\cpy\eda_ai_assist.py" %*
#  setx GEMINI_API_KEY "12345678"
#
# [ site_prompt.txt ]
#   In plain text only.
#   Use US number formatting for large values.
#   Avoid markdown unless explicitly requested.
#   Prefer concise answers and avoid emojis.
#
# @echo off
# python "%ASH_DIR%\ash.py" %*
#######################################
# Linux / macOS
#  export ASH_API_KEY="mykey123"
#  export ASH_MODEL="gemini-2.5-flash"
#  export ASH_PROVIDER="gemini"
#  export ASH_DIR="$HOME/.ash"
#  export ASH_USER_PROMPT="Please address me as Ser Kevin."
# Windows (PowerShell) ash.bat file
#  setx ASH_API_KEY "mykey123"
#  setx ASH_MODEL "gemini-2.5-pro"
#  setx ASH_DIR "C:\Users\Kevin\.ash\logs"
#
# History:
#  2026.02.07 : khubbard : Created, forked from original sump_ai.py. Added CLI
#  2026.02.11 : khubbard : Added paste buffer with threading timeout.
#  2026.02.15 : khubbard : Refactoring. Started to add Azure support (incomplete)
#
# TODO: Consider replacing os.getlogin with getpass.getuser()
########################################################################
ASH_VERSION = "1.0.0"

"""
Cross-platform shell wrapper with history, tab completion, and bash-like bang expansion:
- Windows: PowerShell (prefers pwsh, falls back to powershell.exe)
- Linux/macOS: C shell (csh)

Features:
- Prompt shows current working directory
- Executes commands through the detected shell
- Intercepts `cd` (changes wrapper's working directory)
- Handles exit/quit, Ctrl+C
- Persistent command history saved to ~/.shell_wrapper_history
- Tab completion:
    * POSIX: readline-based completion for files/dirs and executables on PATH
    * Windows: pyreadline3 if available; otherwise history still persists (no completion)
- Bang history expansion (!!, !-n, !n, !prefix) implemented in Python
- Built-in `history` command that prints wrapper history with 1-based indices

Notes:
- Each command runs in a fresh shell, so shell-local state (aliases, env) won't persist.
"""

import os
import sys
import shutil
import subprocess
import signal
import shlex
import glob
import re
from typing import Optional, Literal, Tuple, List
import time, os

# 32‑bit UNIX timestamp (hex)+ 16‑bit CPU PID (hex)+ Original filename
ASH_FILE_RE = re.compile(r"^[0-9a-fA-F]{8}_[0-9a-fA-F]{4}_.+$")

# ---------- Config ----------
PROMPT_COLOR = "\033[36m"
RESET_COLOR = "\033[0m"
HISTORY_FILE = os.path.expanduser("~/.shell_wrapper_history")
HISTORY_LIMIT = 5000
CTRL_G = "\x07"
CTRL_N = "\x0e"

ShellType = Literal["powershell", "csh"]

# ---------- Globals ----------
# In-memory history mirror (always defined). We keep this in sync with readline (if present)
# so bang expansion works everywhere, including Windows without pyreadline3.
INMEM_HISTORY: List[str] = []

# Flags set at runtime by _init_readline()
_readline_available = False
_history_loaded = False
_win_completion_active = False

class api_eda_ai_assist:
    def __init__(self):
        self.provider = None
        self.cfg = None
        self.debug = False

    def open_ai_session( self ):
        if self.debug:
          print("open_ai_session")
        provider_info = self.get_provider_config()
        if provider_info["provider"] == "gemini":
#           self.provider = gemini_provider( self, provider_info["key"], provider_info["model"] )
            self.provider = gemini_provider( self )
        elif provider_info["provider"] == "azure_gateway":
            self.provider = azure_gateway_provider( self )
        else:
            return f"Unknown ai_engine: {provider_info['provider']}"
        self.provider.open_session()

    def close_ai_session( self ):
        self.provider.close_session()

    def is_ai_request(self, prompt):
        import string, shutil, os

        # Tokenize and normalize
        tokens = [t.strip(string.punctuation) for t in prompt.lower().split()]
        if not tokens:
            return False

        first = tokens[0]

        AMBIGUOUS_COMMANDS = {
            "locate", "find", "which", "compare",
            "sort", "split", "join", "write",
        }

        AI_TRIGGERS = {
            "how", "why", "what", "when", "where", "who",
            "explain", "describe", "tell", "show", "help",
            "analyze", "interpret", "summarize", "compare",
            "count", "find", "identify", "measure", "detect",
            "decode", "please", "can", "could", "would",
            "create", "generate",
            "examine", "determine", "are", "ash",
        }

        NATURAL_LEADING_WORDS = {
            "the", "a", "an", "this", "that", "these", "those",
            "my", "your", "our", "their",
        }

        def looks_like_flag(tok: str) -> bool:
            return tok.startswith("-")

        def looks_like_path(tok: str) -> bool:
            return tok.startswith(("/", "./", "../"))

        def looks_like_glob(tok: str) -> bool:
            return any(ch in tok for ch in "*?[")

        def looks_like_filename(tok: str) -> bool:
            # crude but effective: has a dot and no spaces
            return "." in tok and not tok.startswith(".")

        # ------------------------------------------------------------
        # 1. If first word is an executable and NOT ambiguous → shell
        # ------------------------------------------------------------
        if shutil.which(first) is not None and first not in AMBIGUOUS_COMMANDS:
            return False  # shell

        # ------------------------------------------------------------
        # 2. If any obvious AI trigger is present → AI
        # ------------------------------------------------------------
        if any(t in AI_TRIGGERS for t in tokens):
            return True

        # ------------------------------------------------------------
        # 3. Special handling for ambiguous commands
        # ------------------------------------------------------------
        if first in AMBIGUOUS_COMMANDS:
            second = tokens[1] if len(tokens) > 1 else ""

            # Natural-language pattern: "find the ...", "compare the ...", etc.
            if second in NATURAL_LEADING_WORDS:
                return True  # AI

            # Shell-like patterns: flags, paths, globs, filenames
            if (
                looks_like_flag(second)
                or looks_like_path(second)
                or looks_like_glob(second)
                or looks_like_filename(second)
            ):
                return False  # shell

            # If we get here, it's ambiguous; bias toward AI for safety
            return True

        # ------------------------------------------------------------
        # 4. Fallback: no triggers, not ambiguous → assume shell
        # ------------------------------------------------------------
        return False

    def make_ash_cloud_name(self, local_path: str) -> str:
        ts_hex  = f"{int(time.time()) & 0xFFFFFFFF:08x}"
        pid_hex = f"{os.getpid() & 0xFFFF:04x}"
        base    = os.path.basename(local_path)
        return f"{ts_hex}_{pid_hex}_{base}"

    def is_ash_file(name: str) -> bool:
        return bool(ASH_FILE_RE.match(name))

#   def cleanup_cloud_files(client, max_age_seconds=24*3600):
#       now = time.time()
#       removed = 0
#
#       for f in client.files.list():
#           name = f.name
#
#           # 1. Delete anything not created by Ash
#           if not is_ash_file(name):
#               try:
#                   client.files.delete(name=name)
#                   removed += 1
#                   print(f"Deleted foreign cloud file: {name}")
#               except Exception as e:
#                   print(f"Warning: could not delete {name}: {e}")
#               continue
#
#           # 2. Parse timestamp from filename
#           ts_hex = name.split("_", 1)[0]
#           try:
#               ts = int(ts_hex, 16)
#           except ValueError:
#               # malformed → delete it
#               try:
#                   client.files.delete(name=name)
#                   removed += 1
#                   print(f"Deleted malformed cloud file: {name}")
#               except Exception as e:
#                   print(f"Warning: could not delete {name}: {e}")
#               continue
#
#           # 3. Delete if older than max age
#           if now - ts > max_age_seconds:
#               try:
#                   client.files.delete(name=name)
#                   removed += 1
#                   print(f"Deleted stale cloud file: {name}")
#               except Exception as e:
#                   print(f"Warning: could not delete {name}: {e}")
#
#       return removed


    def ask_ai(self, prompt ):
        cfg = self.cfg
        user_prompt = cfg["ASH_USER_PROMPT"]
        log_dir     = cfg["ASH_LOG_DIR"]

        output_file     = self.ai_output_file( prompt )
        input_file_list = self.ai_input_files( prompt, output_file )

        site_prompt = self.load_site_prompt()
        custom_prompt = "\n".join( p for p in [site_prompt, prompt] if p )

#       full_prompt     = self.build_ai_prompt(custom_prompt, input_file_list)
        full_prompt     = custom_prompt

        # Check for really large prompts and ask permission to proceed
#       if full_prompt and not self.warn_if_large(full_prompt): 
#           print("Aborted by user.") 
#           return
#       print("Model = %s" % model )
#       print( full_prompt )

        response_text = "Fake Response"
        upload_bytes = len(full_prompt.encode("utf-8"))
        download_bytes = len(response_text.encode("utf-8"))
        query_log  = os.path.join(log_dir, "usage_queries.log")
        totals_log = os.path.join(log_dir, "usage_totals.log")

        identity = self.get_log_identity();
        ai_engine = cfg["ASH_PROVIDER"]+":"+cfg["ASH_MODEL"]
        api_key = cfg["ASH_API_KEY"]
        self.log_query_usage(query_log,  ai_engine, api_key, upload_bytes, download_bytes, identity )
        self.log_user_totals(totals_log, ai_engine, api_key, upload_bytes, download_bytes, identity )

#       username = os.getlogin()
#       if not self.user_in_usage_totals(username, totals_log ):
#           if not self.require_user_agreement(username, log_dir):
#               return "User declined site restrictions."


#       provider_info = self.get_provider_config()
        result = self.ask_ai_model( prompt, input_file_list )

        if output_file:
            try:
                with open(output_file, "w") as f:
                    f.write(result + "\n")
            except Exception as e:
                print(f"Error writing to {output_file}: {e}")
                print(result)
            else:
                # Still print errors to screen even if writing to file
                if result.startswith("AI error:"):
                    print(result)
        else:
#           print(result)
            return result

    def ask_ai_model( self, prompt, input_file_list ):
        if self.debug:
            print("ask_ai_model() %s" % prompt )
        return self.provider.send_message( prompt, input_file_list );

    def extract_model_override(self, default_model, prompt):
        """
        Detect inline model overrides like:
          "use model gpt-5"
          "using model gpt4-mini"
          "with model gemini-2.5-flash"
          "model llama3-70b"

        Returns:
            (model_to_use, cleaned_prompt)
        """

        text = prompt
        lower = prompt.lower()

        # Find the keyword "model "
        idx = lower.find("model ")
        if idx == -1:
            return default_model, prompt  # no override

        # Extract everything after "model "
        after = lower[idx + len("model "):]

        # Model name is the next token
        parts = after.split()
        if not parts:
            return default_model, prompt  # malformed override
        override = parts[0].strip()

        # Remove the override phrase from the original prompt
        # We remove: "<prefix> model <override>"
        # by reconstructing the exact substring from the original text.
        start = idx
        end = idx + len("model ") + len(override)

        cleaned = (text[:start] + text[end:]).strip()

        # Collapse double spaces
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")

        return override, cleaned

#   def build_ai_prompt(self, user_prompt, input_file_list):
#       parts = [user_prompt.strip(), "", "Attached files:"]
#       for fname in input_file_list:
#           try:
#               with open(fname, "r", encoding="utf-8", errors="replace") as f:
#                   contents = f.read()
#           except OSError:
#               continue  # or log/notify
#           if self.is_binary_file(fname):
#               print(f"Error: {fname} appears to be a binary file.")
#               print("Ash can only process text-based input files.")
#               print("Please provide a decoded or textual representation instead.")
#               return
#
#           parts.append(f"--- BEGIN FILE {fname} ---")
#           parts.append(contents)
#           parts.append(f"--- END FILE {fname} ---")
#       return "\n".join(parts).strip()

#   def warn_if_large(self, full_prompt, threshold_mb=1.0):
#       size_bytes = len(full_prompt.encode("utf-8"))
#       size_mb = size_bytes / (1024 * 1024)
#
#       if size_mb > threshold_mb:
#           print(f"Warning: prompt size is {size_mb:.2f} MB.")
#           print("This may cost dollars rather than cents.")
#           resp = input("Continue? [y/N]: ").strip().lower()
#           return resp in ("y", "yes")
#       return True

#   def is_binary_file(self, filename, blocksize=4096):
#       try:
#           with open(filename, "rb") as f:
#               chunk = f.read(blocksize)
#       except OSError:
#           return False  # treat unreadable as non-binary for now
#
#       # Heuristic: null bytes or too many non-text characters
#       if b"\x00" in chunk:
#           return True
#
#       # Count non-printable bytes
#       text_chars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x7F)))
#       nontext = sum(b not in text_chars for b in chunk)
#
#       return nontext / max(1, len(chunk)) > 0.30


    def obfuscate_key(self, api_key):
        import hashlib
        if not api_key:
            return "none"
        h = hashlib.sha256(api_key.encode()).hexdigest()
        return h[:10]   # short, non‑reversible fingerprint

    def get_log_identity(self):
        """
        Returns the identity string to store in logs based on ASH_LOG_IDENTITY.
        Modes:
            username  → real username
            process   → anonymized process ID
        """
        mode = self.cfg.get("ASH_LOG_IDENTITY", "username").lower()

        if mode == "process":
#           return str(os.getpid())
            return "%08x" % os.getpid()

        # Default: username
        try:
            return os.getlogin()
        except Exception:
            # Fallback if getlogin() fails (cron, systemd, etc.)
            return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"

    def log_query_usage(self, filename, model, api_key, upload_bytes, download_bytes, identity):
        import time
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        key_id = self.obfuscate_key(api_key)
        line = f"{ts}\t{identity}\t{model}\t{key_id}\t{upload_bytes}\t{download_bytes}\n"
        with open( filename, "a") as f:
            f.write(line)


    def log_user_totals(self, filename, model, api_key, upload_bytes, download_bytes, identity):
        import os
        key_id = self.obfuscate_key(api_key)
        totals = {}

        # Load existing totals
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    user = parts[0]
                    data = {kv.split("=")[0]: kv.split("=")[1] for kv in parts[1:]}
                    totals[user] = data

        # Initialize user entry if missing
        if identity not in totals:
            totals[identity] = {
                "uploads": "0",
                "downloads": "0",
                "model": model,
                "key": key_id,
            }

        # Update this user's totals
        totals[identity]["uploads"]   = str(int(totals[identity]["uploads"].replace(",",""))   + upload_bytes)
        totals[identity]["downloads"] = str(int(totals[identity]["downloads"].replace(",","")) + download_bytes)
        totals[identity]["model"] = model
        totals[identity]["key"] = key_id

        # Compute total bytes across all users
        total_bytes_all = 0
        for user, data in totals.items():
            total_bytes_all += int(data["uploads"].replace(",","")) + int(data["downloads"].replace(",",""))

        # Compute percentage for each user
        # Store it temporarily in the dict for sorting and writing
        for user, data in totals.items():
            user_bytes = int(data["uploads"].replace(",","")) + int(data["downloads"].replace(",",""))
            pct = (user_bytes / total_bytes_all * 100) if total_bytes_all > 0 else 0.0
            data["pct"] = pct

        # Sort users by descending percentage
        sorted_users = sorted(totals.items(), key=lambda x: x[1]["pct"], reverse=True)

#       def format_bytes(n):
#           """Convert a byte count into human-readable decimal units."""
#           units = ["B", "KB", "MB", "GB", "TB", "PB"]
#           size = float(n)
#           for unit in units:
#               if size < 1000.0:
#                   return f"{size:.1f}{unit}"
#               size /= 1000.0
#           return f"{size:.1f}EB"

        # Write back sorted totals
        with open(filename, "w") as f:
            for user, data in sorted_users:
                pct_str = f"{data['pct']:.1f}%"
                uploads = int(data['uploads'].replace(",",""))
                downloads = int(data['downloads'].replace(",",""))

                line = ( 
                    f"{user} " 
                    f"pct={pct_str} " 
                    f"uploads={uploads:,} " 
                    f"downloads={downloads:,} " 
                    f"model={data['model']} " 
                    f"key={data['key']}\n"
                )
                f.write(line)

#   def user_in_usage_totals(self, username, log_path):
#       """
#       Returns True if the user already appears in usage_totals.log.
#       Otherwise returns False.
#       """
#       try:
#           with open(log_path, "r") as f:
#               for line in f:
#                   if line.strip().startswith(username + " "):
#                       return True
#       except FileNotFoundError:
#           # No log yet → no users recorded
#           return False
#
#       return False

#   def require_user_agreement(self, username, ash_dir):
#       """
#       If the user is new, display site_restrictions.txt (if present)
#       and ask for confirmation. Returns True if the user agrees,
#       False if they decline.
#       """
#
#       restrictions_path = os.path.join(ash_dir, "site_restrictions.txt")
#
#       # If no restrictions file exists, auto-approve
#       if not os.path.exists(restrictions_path):
#           return True
#
#       # Display restrictions
#       print("\n---------------- SITE RESTRICTIONS ----------------")
#       try:
#           with open(restrictions_path, "r") as f:
#               print(f.read().strip())
#       except Exception as e:
#           print(f"(Warning: could not read site_restrictions.txt: {e})")
#       print("---------------------------------------------------\n")
#
#       # Ask for confirmation
#       reply = input("Do you agree and wish to proceed? (yes/no): ").strip().lower()
#
#       if reply in ("yes", "y"):
#           return True
#
#       print("Request cancelled. You must agree to the site restrictions to continue.")
#       return False


    def ai_output_file(self, prompt):
        """
        Parse an output filename from a natural‑language request.
        Returns the filename as a string, or None if not found.

        Examples:
            "output to file foo.txt"          → "foo.txt"
            "write to bar.vcd?"               → "bar.vcd"
            "output to the file results.vcd." → "results.vcd"
            "write to a file named test.vcd"  → "named" (kept simple by design)
        """
        text = prompt.lower()
        TRIGGERS = ("output to", "write to")
        SKIP = {"file", "the", "a"}

        def strip_trailing_punct(tok: str) -> str:
            # Only remove punctuation that cannot be part of a filename
            return tok.rstrip(".,;:!?")

        for trig in TRIGGERS:
            if trig in text:
                after = text.split(trig, 1)[1].strip()

                # Normalize punctuation spacing
                tokens = after.replace(",", " ").replace(";", " ").split()

                for token in tokens:
                    if token not in SKIP:
                        clean = strip_trailing_punct(token)
                        return clean

        return None


    def ai_input_files(self, prompt, out_file ):
        """
        Detect one or more input files in a natural-language request.
        Returns a list of filenames that actually exist, excluding any
        output file detected by ai_output_file().
        """
        text = prompt.lower()
        TRIGGERS = ("file", "files", "analyze", "load")
        SKIP = {"the", "a", "an"}

        # make sure "file foo.txt?" is "foo.txt"
        def strip_punctuation(tok: str) -> str: 
            return tok.rstrip(".,;:!?")

        cleaned = (
            text.replace(",", " ")
                .replace(";", " ")
                .replace(":", " ")
        )

        tokens = cleaned.split()
        found = []

        for i, tok in enumerate(tokens):
            if tok in TRIGGERS:
                for nxt in tokens[i+1:]:
                    if nxt in TRIGGERS:
                        break
                    if nxt in SKIP:
                        continue

                    candidate = strip_punctuation(nxt)

                    if os.path.exists(candidate):
                        # Exclude the output file if present
                        if out_file is None or candidate != out_file:
                            found.append(candidate)
        return found


    def load_site_prompt(self):
        base = os.environ.get("ASH_DIR", os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "site_prompt.txt")
        if not os.path.exists(path):
            return ""
        with open(path, "r") as f:
            return f.read().strip()

    def load_site_secret_key(self):
        """
        Loads the site secret key from $ASH_DIR/site_key.txt.
        Returns the key as a string, or None if missing or unreadable.
        """

        ash_dir = os.environ.get("ASH_DIR")
        if not ash_dir:
            print("Error: ASH_DIR environment variable is not set.")
            return None

        key_path = os.path.join(ash_dir, "site_key.txt")

        try:
            with open(key_path, "r") as f:
                key = f.read().strip()
                if not key:
                    print("Error: site_key.txt is empty.")
                    return None
                return key

        except FileNotFoundError:
            print(f"Error: site_key.txt not found in {ash_dir}.")
            return None

        except Exception as e:
            print(f"Error reading site_key.txt: {e}")
            return None


    def get_provider_config(self):
        if self.cfg is None:
            self.get_env_config()
        cfg = self.cfg
        provider = cfg["ASH_PROVIDER"]
        endpoint = cfg["ASH_ENDPOINT"]
        model    = cfg["ASH_MODEL"]
        key      = cfg["ASH_API_KEY"] 
        return {
            "provider": provider,
            "key": key,
            "endpoint": endpoint,
            "model": model,
        }
 
    def get_env_config(self):
        if self.cfg is not None:
            return self.cfg
        """
        Load configuration in this priority order:
            1. Internal Python defaults
            2. User OS environment variable for ASH_DIR
            3. Site defaults from $ASH_DIR/site_defaults.txt
            4. User OS environment variables for any ASH_* key
        """

        # ------------------------------------------------------------
        # 1. Internal defaults
        # ------------------------------------------------------------
        cfg = {
            "ASH_DIR": os.path.expanduser("~/.ash"),
            "ASH_PROVIDER": "gemini",
            "ASH_ENDPOINT": "https://generativelanguage.googleapis.com/v1beta/openai",
            "ASH_TOKEN": "",
            "ASH_API_KEY": "",
            "ASH_API_VERSION": "",
            "ASH_MODEL": "gemini-2.0-flash",
            "ASH_USER_PROMPT": "",
            "ASH_LOG_IDENTITY": "username",
            "ASH_LOG_DIR": None,   # will default to ASH_DIR later
        }

        # ------------------------------------------------------------
        # 2. User OS environment variable for ASH_DIR (special case)
        # ------------------------------------------------------------
        if "ASH_DIR" in os.environ:
            cfg["ASH_DIR"] = os.environ["ASH_DIR"]

        ash_dir = cfg["ASH_DIR"]

        # ------------------------------------------------------------
        # 3. Load site defaults from $ASH_DIR/site_defaults.txt
        # ------------------------------------------------------------
        site_defaults = {}
        defaults_path = os.path.join(ash_dir, "site_defaults.txt")

        if os.path.exists(defaults_path):
            try:
                with open(defaults_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip()
                            if val.startswith('"') and val.endswith('"'):
                                val = val[1:-1]
                            site_defaults[key] = val
            except Exception as e:
                print(f"Warning: could not read site_defaults.txt: {e}")

        # Apply site defaults (override internal defaults)
        for key, val in site_defaults.items():
            cfg[key] = val

        # ------------------------------------------------------------
        # 4. User OS environment variables override everything
        # ------------------------------------------------------------
        for key in cfg.keys():
            if key in os.environ:
                cfg[key] = os.environ[key]

        # ------------------------------------------------------------
        # 5. Use Token or API_KEY 
        #    Note: Gemini may default to GEMINI_API_KEY
        # ------------------------------------------------------------
        ash_token = cfg["ASH_TOKEN"]
        ash_key   = cfg["ASH_API_KEY"]
        if ash_token:
            secret_key = self.load_site_secret_key();
            username, key = self.decrypt_token( secret_key, ash_token )
            if username == os.getlogin():
                cfg["ASH_API_KEY"] = key


        # ------------------------------------------------------------
        # Finalize dependent defaults
        # ------------------------------------------------------------
        if not cfg["ASH_LOG_DIR"]:
            cfg["ASH_LOG_DIR"] = cfg["ASH_DIR"]

        self.cfg = cfg


    def decrypt_token( self, secret_key, token ):
        import hmac, hashlib

        try:
            username, cipher_hex, sig = token.split("|")
        except ValueError:
            return None, None

        key_bytes = secret_key.encode()
        payload = f"{username}|{cipher_hex}"
        expected = hmac.new(key_bytes, payload.encode(), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected, sig):
            return None, None  # tampered or invalid

        cipher = bytes.fromhex(cipher_hex)

        def xor_bytes(data, key):
            key = key * (len(data) // len(key) + 1)
            return bytes([a ^ b for a, b in zip(data, key)])

        api_bytes = xor_bytes(cipher, key_bytes)
        return username, api_bytes.decode()


# Abstract base provider class. This defines the contract.
# Every provider must implement these two methods.
class ai_provider:
    def open_session(self):
        raise NotImplementedError

    def close_session(self):
        raise NotImplementedError

    def send_message(self, prompt, input_file_list):
        raise NotImplementedError

class azure_gateway_provider(ai_provider):
    def __init__(self, parent):
        self.parent = parent
        self.client = None
        self.chat = None
        self.api_key = parent.cfg["ASH_API_KEY"]
        self.model = parent.cfg["ASH_MODEL"]
        self.session_file_list = []
        self.debug = parent.debug

    def open_session( self ):
        if self.debug:
            print("open_session(azure_gateway:%s)" % self.model )
#       from openai import AzureOpenAI
#       self.client = AzureOpenAI( azure_endpoint= self.parent.cfg["ASH_ENDPOINT"],
#                                  api_key=        self.parent.cfg["ASH_API_KEY"],
#                                  api_version=    self.parent.cfg["ASH_API_VERSION"] )
#       self.chat = self.client.chat
 
        self.intro   = [ {"role": "system", "content": "You are a concise, helpful assistant."} ]
        self.history = []
 
    def close_session( self ):
        if self.debug:
            print("close_session(azure_gateway)")

    def send_message(self, prompt, input_file_list ):
        message = []
        file_blocks = []
        file_message = None

        # If a file is in the input_file_list and NOT in the session_file_list, upload it and add to session list
        for each_file in input_file_list:
            if not any( each_file == item[0] for item in self.session_file_list):
                name = self.parent.make_ash_cloud_name( each_file ) # Timestamp+CPU_PID+each_name
                self.session_file_list.append( (each_file, name ))
        for each_file, name in self.session_file_list:
            if self.debug:
                print("uploading %s" % each_file )
            with open( each_file, "r", encoding="utf-8") as f:
                content = f.read()
            file_blocks.append(f"=== FILE: {os.path.basename(each_file)} ===\n{content}\n")
        files_context = "\n".join(file_blocks)

        if self.session_file_list:
          file_message = f"Here are my files:\n\n{files_context}\n\nAnswer questions using these files and referencing their filenames."

        if file_message:
            message.append({"role": "user", "content": file_message })
        message += [ self.intro ]
        message += self.history
        prompt_message = {"role": "user", "content": prompt}
#       response = self.chat.completions.create( model=self.model, messages= message )
#       answer = response.choices[0].message.content
        if self.debug:
            print( message )
        answer = "A:" + str( prompt )
        self.history.append( prompt_message )
        self.history.append({"role": "assistant", "content": answer})
        return answer


class gemini_provider(ai_provider):
    def __init__(self, parent):
        self.parent = parent
        self.client = None
        self.chat = None
        self.api_key = parent.cfg["ASH_API_KEY"]
        self.model = parent.cfg["ASH_MODEL"]
        self.session_file_list = []
        self.debug = parent.debug

    def open_session( self ):
        if self.debug:
            print("open_session(gemini:%s)" % self.model )
        from google import genai
 
        # If api_key is None, the client auto‑reads GEMINI_API_KEY
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client()  # auto-reads GEMINI_API_KEY
        self.chat = self.client.chats.create(model=self.model)

    def close_session( self ):
        if self.debug:
            print("close_session(gemini)")
            print("Current Files:")
        for file in self.client.files.list():
            # Name: files/2qgse62no0hh | Display Name: None | URI: https://generativelanguage.googleapis.com/v1beta/files/2qgse62no0hh
            print(f"- Name: {file.name} | Display Name: {file.display_name} | URI: {file.uri}")
            try:
                self.client.files.delete(name = file.name )
                if self.debug:
                    print("File deleted successfully.")
            except:
                if self.debug:
                    print("Failed to delete.")

    def send_message(self, prompt, input_file_list ):
        if self.debug:
            print("send_message(gemini) %s" % prompt )
        client = self.client
        chat = self.chat
        session_file_list = self.session_file_list

        # If a file is in the input_file_list and NOT in the session_file_list, upload it and add to session list
        for each_file in input_file_list:
            if not any( each_file == item[0] for item in session_file_list):
                if self.debug:
                    print(f"Attempting to Upload file '{each_file}'")
                # 1. Upload each file using the Files API
                name = self.parent.make_ash_cloud_name( each_file ) # Timestamp+CPU_PID+each_name
                uploaded_file = client.files.upload(file=each_file, config={'mime_type':'text/plain', 'display_name':name})
                if self.debug:
                    print(f"Uploaded file '{each_file}' as: {uploaded_file.name}") # Uploaded file 'foo.txt' as: files/bd0blct1aolk
                session_file_list.append( (each_file, uploaded_file ))

#       # 2. Create the contents list for the prompt using only files in the input_file_list
#       # It's important to add descriptive text to help the model distinguish between files
        contents = []; i = 0;
        if input_file_list:
            contents.append("Here are files for analysis")
            for each_file, uploaded_file in session_file_list:
                if each_file in input_file_list:
                    contents.append(f"File {i+1} ({os.path.basename( each_file )}):")
                    contents.append(uploaded_file)
        contents.append(prompt)

#       print("Current Files:")
#       for file in client.files.list():
#           # Name: files/2qgse62no0hh | Display Name: None | URI: https://generativelanguage.googleapis.com/v1beta/files/2qgse62no0hh
#           print(f"- Name: {file.name} | Display Name: {file.display_name} | URI: {file.uri}")

        # print( contents )
        #['Content of File 1 (foo.txt):', File(
        #  create_time=datetime.datetime(2026, 2, 12, 18, 40, 47, 491241, tzinfo=TzInfo(0)),
        #  display_name='foo.txt',
        #  expiration_time=datetime.datetime(2026, 2, 14, 18, 40, 47, 20205, tzinfo=TzInfo(0)),
        #  mime_type='text/plain',
        #  name='files/t9oczyrv66u3',
        #  sha256_hash='YTg4M2RhZmM0ODBkNDY2ZWUwNGUwZDZkYTk4NmJkNzhlYjFmZGQyMTc4ZDA0NjkzNzIzZGEzYThmOTVkNDJmNA==',
        #  size_bytes=5,
        #  source=<FileSource.UPLOADED: 'UPLOADED'>,
        #  state=<FileState.ACTIVE: 'ACTIVE'>,
        #  update_time=datetime.datetime(2026, 2, 12, 18, 40, 47, 491241, tzinfo=TzInfo(0)),
        #  uri='https://generativelanguage.googleapis.com/v1beta/files/t9oczyrv66u3'
        #), 'Content of File 2 (bar.txt):', File(
        #    ...
        #
        #)]

        try:
            response = chat.send_message( contents )
            if self.debug:
                print("chat.send_message(gemini) \nQ: %s \nA: %s" % ( contents, response.text.strip() ) )
            return response.text.strip()
        except Exception as e:
            return f"AI error: {type(e).__name__}: {e}"


# ---------- Version ----------
def print_version( self ):
    import textwrap

    self.get_env_config()
    cfg = self.cfg

    version_text = f"""
    Ash — AI‑Enabled EDA Assistant
    Version: {ASH_VERSION}

    Ash is a command‑line tool for natural‑language analysis of EDA files.
    It operates using a site‑assigned API key and a configurable AI model.

    Current Configuration
      ASH_DIR:          {cfg.get("ASH_DIR")}
      ASH_PROVIDER:     {cfg.get("ASH_PROVIDER")}
      ASH_MODEL:        {cfg.get("ASH_MODEL")}
      ASH_ENDPOINT:     {cfg.get("ASH_ENDPOINT")}
      ASH_API_VERSION:  {cfg.get("ASH_API_VERSION")}
      ASH_LOG_DIR:      {cfg.get("ASH_LOG_DIR")}
      ASH_LOG_IDENTITY: {cfg.get("ASH_LOG_IDENTITY")}

    License
      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, version 3 or later.

    Authors
      Kevin Hubbard — Black Mesa Labs
      Additional engineering assistance provided by Microsoft Copilot.
    """

    print(textwrap.dedent(version_text).rstrip())


# ---------- Help Manual ----------
def print_help(self):
    import textwrap

    self.get_env_config()
    cfg = self.cfg

    help_text = f"""
    Usage: ash [options] [command]

    Ash is a command‑line assistant for natural‑language analysis of EDA files.
    It can execute shell commands or interpret plain‑English requests using the
    configured AI provider.

    Options:
      -h, --help        Show this help message and exit
      -v, --version     Show version and configuration information

    Shell Behavior:
      • Commands that match executables run in the system shell
      • Natural‑language requests are routed to the AI engine
      • Use Ctrl+D on an empty line to enter or exit multi‑line mode
      • History, bang expansion, and tab completion are supported

    AI Configuration:
      Provider:    {cfg.get("ASH_PROVIDER")}
      Model:       {cfg.get("ASH_MODEL")}
      Endpoint:    {cfg.get("ASH_ENDPOINT")}
      API Version: {cfg.get("ASH_API_VERSION")}
      API Key:     {'<set>' if cfg.get('ASH_API_KEY') else '<not set>'}

    File Locations:
      ASH_DIR:    {cfg.get("ASH_DIR")}
      Log Dir:    {cfg.get("ASH_LOG_DIR")}
      Identity:   {cfg.get("ASH_LOG_IDENTITY")}  (username or process)

    Environment Variables:
      ASH_DIR            Base directory for site configuration
      ASH_PROVIDER       AI provider name (default: gemini)
      ASH_MODEL          Model name for the provider
      ASH_ENDPOINT       Provider API endpoint
      ASH_API_VERSION    Provider API version
      ASH_API_KEY        Raw API key (optional if ASH_TOKEN is used)
      ASH_TOKEN          Encrypted API token (site‑managed)
      ASH_USER_PROMPT    Optional prefix added to all AI requests
      ASH_LOG_DIR        Directory for usage logs
      ASH_LOG_IDENTITY   'username' or 'process'

    Examples:
      ash
      ash "summarize file foo.vcd"
      ash "how many lines are in bar.txt?"
      ash "compare files foo.vcd bar.vcd"
    """

    print(textwrap.dedent(help_text).rstrip())

    # Optional site files
    ash_dir = cfg.get("ASH_DIR")
    billing_path = os.path.join(ash_dir, "site_billing.txt") if ash_dir != "<not set>" else None
    restrictions_path = os.path.join(ash_dir, "site_restrictions.txt") if ash_dir != "<not set>" else None

    billing_text = ""
    restrictions_text = ""

    # Append optional sections
    if billing_text:
        print("\nBilling Information (site_billing.txt)")
        print("--------------------------------------")
        print(billing_text)

    if restrictions_text:
        print("\nSite Restrictions (site_restrictions.txt)")
        print("-----------------------------------------")
        print(restrictions_text)
    print()


# ---------- Platform Detection ----------
def on_windows() -> bool:
    return sys.platform.startswith("win32")

# ---------- Shell Discovery ----------
def find_powershell() -> Optional[str]:
    # Prefer PowerShell 7+ (pwsh), then Windows PowerShell
    for candidate in ("pwsh", "powershell.exe", "powershell"):
        path = shutil.which(candidate)
        if path:
            return path
    # Common fallbacks
    for candidate in (r"C:\Program Files\PowerShell\7\pwsh.exe",
                      r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"):
        if os.path.exists(candidate):
            return candidate
    return None

def find_csh() -> Optional[str]:
    path = shutil.which("csh")
    if path:
        return path
    for candidate in ("/bin/csh", "/usr/bin/csh"):
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None

def detect_shell() -> Tuple[ShellType, str]:
    if on_windows():
        ps = find_powershell()
        if not ps:
            print("Error: PowerShell not found (tried pwsh and powershell.exe).", file=sys.stderr)
            sys.exit(127)
        return "powershell", ps
    else:
        csh_path = find_csh()
        if not csh_path:
            print("Error: `csh` not found. Please install C shell (package `csh`) and try again.", file=sys.stderr)
            sys.exit(127)
        return "csh", csh_path

# ---------- Prompt ----------
def format_prompt(buffering_ai, buffering_ai_ctrl_d_hint_sent ) -> str:
    if buffering_ai:
      prompt = "...> "
      if not buffering_ai_ctrl_d_hint_sent:
        prompt = "(Ctrl+D to send)\n"+prompt
      return prompt
    cwd = truncate_string(os.getcwd())
#   return f"{PROMPT_COLOR}{cwd}{RESET_COLOR} % "
    return f"[ash]:{cwd}% "

def truncate_string(input_string, max_length=30):
    # Truncates a string to the last 'max_length' characters if it's longer.
    if len(input_string) > max_length:
        return input_string[-max_length:]
    else:
        return input_string


# ---------- Built-in cd ----------
def handle_cd(arg_str: str) -> None:
    target = arg_str.strip() or os.path.expanduser("~")
    target = os.path.expandvars(os.path.expanduser(target))
    try:
        os.chdir(target)
    except FileNotFoundError:
        print(f"cd: no such file or directory: {target}", file=sys.stderr)
    except NotADirectoryError:
        print(f"cd: not a directory: {target}", file=sys.stderr)
    except PermissionError:
        print(f"cd: permission denied: {target}", file=sys.stderr)
    except Exception as exc:
        print(f"cd: {exc}", file=sys.stderr)

# ---------- Command Execution ----------
def run_shell_command(shell_type: ShellType, shell_path: str, command: str) -> int:
    if shell_type == "powershell":
        argv = [shell_path, "-NoLogo", "-NoProfile", "-Command", command]
    else:
        argv = [shell_path, "-c", command]
    try:
        completed = subprocess.run(
            argv,
            stdin=None,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=os.getcwd(),
            env=os.environ.copy(),
            text=False,
            check=False
        )
        return completed.returncode
    except FileNotFoundError:
        print(f"Error: shell not found while executing: {shell_path}", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        print(f"Execution error: {exc}", file=sys.stderr)
        return 1

# ---------- History & Completion ----------
def get_path_executables() -> List[str]:
    """Return a list of executable names found in PATH (basename only)."""
    exes = set()
    path_env = os.environ.get("PATH", "")
    sep = ";" if on_windows() else ":"
    for p in path_env.split(sep):
        if not p or not os.path.isdir(p):
            continue
        try:
            for name in os.listdir(p):
                full = os.path.join(p, name)
                if os.path.isfile(full) and os.access(full, os.X_OK):
                    exes.add(name)
        except Exception:
            continue
    return sorted(exes)

def _get_history_list() -> List[str]:
    """
    Return a list of history lines in order (oldest -> newest).
    Use readline if available; otherwise return INMEM_HISTORY.
    """
    if _readline_available:
        try:
            import readline  # type: ignore
            return [readline.get_history_item(i + 1)
                    for i in range(readline.get_current_history_length())]
        except Exception:
            pass
    return INMEM_HISTORY[:]

def _init_readline():
    """
    Initialize readline/pyreadline3 if available.
    - Load history file into readline AND INMEM_HISTORY
    - Configure tab completion
    """
    global _readline_available, _history_loaded, _win_completion_active
    _readline_available = False
    _history_loaded = False
    _win_completion_active = False

    # Try to import readline (POSIX) or pyreadline3's readline (Windows)
    try:
        import readline  # type: ignore
        _readline_available = True
    except Exception:
        _readline_available = False

    if _readline_available:
        import readline  # type: ignore

        # Load history file, if present
        try:
            if os.path.exists(HISTORY_FILE):
                readline.read_history_file(HISTORY_FILE)
                _history_loaded = True
        except Exception:
            pass

        # Sync readline's history into our in-memory mirror
        try:
            INMEM_HISTORY.clear()
            for i in range(readline.get_current_history_length()):
                item = readline.get_history_item(i + 1)
                if item is not None:
                    INMEM_HISTORY.append(item)
        except Exception:
            pass

        # Set history length
        try:
            readline.set_history_length(HISTORY_LIMIT)
        except Exception:
            pass

        # Configure tab completion
        try:
            executables_cache = get_path_executables()

            def complete(text, state):
                buffer = readline.get_line_buffer()
                cursor = readline.get_endidx()
                try:
                    lex = shlex.split(buffer[:cursor])
                    at_start = (len(lex) == 0) or (
                        buffer.strip()
                        and buffer.strip() == lex[0]
                        and cursor <= len(lex[0])
                    )
                except Exception:
                    parts = buffer[:cursor].split()
                    at_start = (len(parts) <= 1)

                candidates: List[str] = []
                if at_start:
                    candidates.extend(executables_cache)

                pattern = os.path.expandvars(os.path.expanduser(text)) + "*"
                matches = glob.glob(pattern)
                for m in matches:
                    display = m
                    if os.path.isdir(m) and not display.endswith(os.sep):
                        display += os.sep
                    candidates.append(display)

                # Deduplicate while preserving order; filter by startswith(text)
                seen = set()
                ordered = []
                for c in candidates:
                    if c not in seen and c.startswith(text):
                        seen.add(c)
                        ordered.append(c)

                if state < len(ordered):
                    return ordered[state]
                return None

            readline.set_completer(complete)
            if on_windows():
                _win_completion_active = True
            else:
                readline.parse_and_bind("tab: complete")
        except Exception:
            pass

    else:
        # No readline: preload existing history file into INMEM_HISTORY (if any)
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if line:
                            INMEM_HISTORY.append(line)
                _history_loaded = True
        except Exception:
            pass

def _save_history():
    """
    Persist history to HISTORY_FILE.
    If readline is available, let it write the file.
    Otherwise, write INMEM_HISTORY ourselves (preserve order, truncate to limit).
    """
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    except Exception:
        pass

    if _readline_available:
        try:
            import readline  # type: ignore
            readline.set_history_length(HISTORY_LIMIT)
            readline.write_history_file(HISTORY_FILE)
            return
        except Exception:
            pass  # fall through

    # Manual write from INMEM_HISTORY (preserve order)
    try:
        start = max(0, len(INMEM_HISTORY) - HISTORY_LIMIT)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            for item in INMEM_HISTORY[start:]:
                f.write(item + "\n")
    except Exception:
        pass

# ---------- Bang expansion ----------
BANG_RE = re.compile(r"^!(.+)$")

def expand_bang(line: str) -> str:
    """
    Perform bash-like history expansion on a single line.
    Supported:
      !!      -> last command
      !-n     -> nth previous command (e.g., !-2)
      !n      -> absolute command number (1-based)
      !prefix -> most recent command starting with 'prefix'
    Returns expanded line; raises ValueError on failure.
    """
    m = BANG_RE.match(line.strip())
    if not m:
        return line

    token = m.group(1).strip()
    hist = _get_history_list()
    if not hist:
        raise ValueError("event not found: history is empty")

    if token == "!":            # '!!'
        return hist[-1]

    if token.startswith("-"):   # '!-n'
        try:
            n = int(token[1:])
            if n <= 0 or n > len(hist):
                raise ValueError
        except Exception:
            raise ValueError(f"bad event specification: !{token}")
        return hist[-n]

    if token.isdigit():         # '!n' (1-based index into wrapper history)
        idx = int(token)
        if idx <= 0 or idx > len(hist):
            raise ValueError(f"event not found: !{token}")
        return hist[idx - 1]

    # '!prefix' -> last command starting with prefix
    prefix = token
    for cmd in reversed(hist):
        if cmd.startswith(prefix):
            return cmd

    raise ValueError(f"event not found: !{token}")

# ---------- Built-in 'history' ----------
def print_history():
    """Print wrapper history with 1-based indices (oldest → newest)."""
    hist = _get_history_list()
    for i, cmd in enumerate(hist, start=1):
        print(f"{i:5d}  {cmd}")

# --- Main REPL (Read-Eval-Print Loop) ---
def main():
    ai = api_eda_ai_assist()
    shell_type, shell_path = detect_shell()
    provider = None
    buffering_ai_ctrl_d_hint_sent = False
    buffering_ai = False
    paste_buffer = []  
    BLOCK_INTRO_RE = None
    full_prompt = None

    _init_readline()
    if on_windows() and not _readline_available:
        print("Note: Tab completion not available. Install `pyreadline3` (pip install pyreadline3) for Windows completion.")

    # Let child processes receive SIGINT; keep wrapper alive
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    if "--help" in sys.argv or "-h" in sys.argv:
        print_help( ai )
        return

    if "--version" in sys.argv or "-v" in sys.argv:
        print_version( ai )
        return


    # If arguments were passed, run in "one-shot" mode 
    if len(sys.argv) > 1: # Example: python ash.py analyze foo.txt 
        line = " ".join(sys.argv[1:]) 
        api_key = None
        ai.open_ai_session();
        rts = ai.ask_ai( line )
        print( rts )
        if ai.provider:
            ai.close_ai_session();
        return

#   banner = f"{shell_type} wrapper (using {shell_path}). Type `history`, `exit`, or press Ctrl+C to quit."
#   print(banner)
#                                                                                         #  
    print("------------------------------------------------------------------------")
    print("Hi, I'm Ash (eda_ai_assist), your cloud‑based AI EDA assistant.         ")
    print("From your shell, I interpret plain‑English and analyze your EDA files.  ")
    print("I became operational at Black Mesa Labs on February 8th, 2026.          ")
    print("Press Ctrl+D on an empty line to enter or exit multi‑line input mode.   ")
    print("------------------------------------------------------------------------")


    while True:
        try:
            try:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
            except Exception:
                pass
            try:
                line = input(format_prompt(buffering_ai,buffering_ai_ctrl_d_hint_sent ))
                if buffering_ai:
                  buffering_ai_ctrl_d_hint_sent = True
                else:
                  buffering_ai_ctrl_d_hint_sent = False
            except EOFError:
               # Ctrl-D pressed on an empty line
                if not buffering_ai:
                    # Enter buffering mode
                    buffering_ai = True
                    paste_buffer = []
                    print("(Begin Buffering. Type your prompt and paste your data. Ctrl+D again to finish.)")
                    continue
                else:
                    # Exit buffering mode and send
                    buffering_ai = False
                    full_prompt = "\n".join(paste_buffer)
                    paste_buffer = []
                    if not ai.provider:
                        print("ai.open_ai_session()")
                        ai.open_ai_session()
                    rts = ai.ask_ai(full_prompt)
                    print(rts)
                    continue

            if not line.strip():
                continue

        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue

        # If we are already buffering an AI request, ALL lines go to the buffer 
        if buffering_ai: 
            paste_buffer.append(line) 
            continue

        # Mirror raw input into in-memory history so bang expansion sees it
        if line.strip():
            INMEM_HISTORY.append(line)
            # Also push into readline history (if available)
            if _readline_available:
                try:
                    import readline  # type: ignore
                    readline.add_history(line)
                except Exception:
                    pass

        # Ignore SIGINT while executing; child receives it
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass

        stripped = line.strip()

        # Exit conditions
        if stripped in {"exit", "quit"} :
            break

        # Built-in 'history' (show wrapper's numbering)
        if stripped == "history" and not buffering_ai:
            print_history()
            continue

        # Bang expansion (before cd / external execution)
        try:
            expanded = expand_bang(line)
            if expanded != line:
                print(expanded)  # echo like bash does
            line = expanded
            stripped = line.strip()
        except ValueError as e:
            print(str(e))
            continue

        # Built-in cd
        if stripped.startswith("cd") :
            parts = stripped.split(maxsplit=1)
            arg = parts[1] if len(parts) == 2 else ""
            handle_cd(arg)
            continue

        # Normal single-line AI request
        if ai.is_ai_request(line):
            if not ai.provider:
                print("ai.open_ai_session()")
                ai.open_ai_session()
            rts = ai.ask_ai(line)
            print(rts)
            continue
         
        # Delegate to the detected shell
        print("OS_CALL: %s" % line )
        _ = run_shell_command(shell_type, shell_path, line)
        

    if ai.provider:
        ai.close_ai_session();

    _save_history()
    print("Bye.")

if __name__ == "__main__":
    main()
