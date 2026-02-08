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
#  export ASH_MODEL="gemini-2.0-pro"
#  export ASH_PROVIDER="gemini"
#  export ASH_DIR="$HOME/.ash"
#  export ASH_USER_PROMPT="Address me as Sir Kevin."
# Windows (PowerShell)
#  setx ASH_API_KEY "mykey123"
#  setx ASH_MODEL "gemini-2.0-pro"
#  setx ASH_DIR "C:\Users\Kevin\.ash\logs"
#
# History:
#  2026.02.07 : Created, forked from original sump_ai.py. Added CLI
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
- Handles exit/quit, Ctrl+C, and Ctrl+D
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

# ---------- Config ----------
PROMPT_COLOR = "\033[36m"
RESET_COLOR = "\033[0m"
HISTORY_FILE = os.path.expanduser("~/.shell_wrapper_history")
HISTORY_LIMIT = 5000

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
    def is_ai_request(self, prompt):
        import string;
        # Removed "which" as this is a Linux command
        AI_TRIGGERS = { "how", "why", "what", "when", "where", "who", "explain",
                    "describe", "tell", "show", "help", "analyze", "interpret",
                    "summarize", "compare", "count", "find", "identify", "measure",
                    "detect", "decode", "please", "can", "could", "would","examine",
                    "ash" }
        tokens = [t.strip(string.punctuation) for t in prompt.lower().split()]
        return any(t in AI_TRIGGERS for t in tokens)


    def ask_ai(self, prompt, ai_engine="gemini", api_key=None):
#       api_key, model, user_prompt, logdir = self.get_env_config()
        user_prompt, logdir = self.get_env_config()

#       self.get_provider_config()
#           return {
#               "provider": "gemini",
#               "key": key,
#               "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
#               "model": os.getenv("ASH_MODEL", "gemini-2.0-flash"),
#           }

        provider_info = self.get_provider_config()

        # User might ask for a different model in the prompt
        model, prompt = self.extract_model_override(provider_info["model"], prompt )

        output_file     = self.ai_output_file( prompt )
        input_file_list = self.ai_input_files( prompt, output_file )

        site_prompt = self.load_site_prompt()
        custom_prompt = "\n".join( p for p in [site_prompt, prompt] if p )

        full_prompt     = self.build_ai_prompt(custom_prompt, input_file_list)

        # Check for really large prompts and ask permission to proceed
        if full_prompt and not self.warn_if_large(full_prompt): 
            print("Aborted by user.") 
            return
        print("Model = %s" % model )
        print( full_prompt )

        response_text = "Fake Response"
        upload_bytes = len(full_prompt.encode("utf-8"))
        download_bytes = len(response_text.encode("utf-8"))
        query_log  = os.path.join(logdir, "usage_queries.log")
        totals_log = os.path.join(logdir, "usage_totals.log")

        self.log_query_usage(query_log, ai_engine, api_key, upload_bytes, download_bytes)
        self.log_user_totals(totals_log, ai_engine, api_key, upload_bytes, download_bytes)

        username = os.getlogin()
        if not self.user_in_usage_totals(username, totals_log ):
            if not self.require_user_agreement(username, logdir):
                return "User declined site restrictions."


#       ai_engine = ai_engine.lower();
#       result = self.ask_ai_model( prompt, ai_engine, api_key )
        result = self.ask_ai_model( prompt, provider_info )

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
            print(result)

    def ask_ai_model( self, prompt, provider_info ):
        if provider_info["provider"] == "gemini":
            return self.ask_gemini(prompt, provider_info["key"], provider_info["model"] )
        else:
            return f"Unknown ai_engine: {provider_info['provider']}"

    def ask_gemini(self, prompt, api_key, model ):
        from google import genai
        # If api_key is None, the client auto‑reads GEMINI_API_KEY
#       client = genai.Client() if api_key is None else genai.Client(api_key=api_key)
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()  # auto-reads GEMINI_API_KEY

        try:
            response = client.models.generate_content( model=model, contents=prompt)
#           response = client.models.generate_content( model="gemini-2.5-flash", contents=prompt)
#           response = client.models.generate_content( model="gemini-2.5-flash-lite", contents=prompt)
            return response.text.strip()
        except Exception as e:
            return f"AI error: {type(e).__name__}: {e}"

#   def ask_ai_model( self, prompt, ai_engine, api_key ):
#       if ai_engine == "gemini":
#           return self.ask_gemini(prompt, api_key)
#       else:
#           return f"Unknown ai_engine: {ai_engine}"
#       totals[username]["uploads"]   = str(int(totals[username]["uploads"].replace(",",""))   + upload_bytes)
#       self.get_provider_config()
#           return {
#               "provider": "gemini",
#               "key": key,
#               "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
#               "model": os.getenv("ASH_MODEL", "gemini-2.0-flash"),
#           }

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



    def build_ai_prompt(self, user_prompt, input_file_list):
        parts = [user_prompt.strip(), "", "Attached files:"]
        for fname in input_file_list:
            try:
                with open(fname, "r", encoding="utf-8", errors="replace") as f:
                    contents = f.read()
            except OSError:
                continue  # or log/notify
            if self.is_binary_file(fname):
                print(f"Error: {fname} appears to be a binary file.")
                print("Ash can only process text-based input files.")
                print("Please provide a decoded or textual representation instead.")
                return

            parts.append(f"--- BEGIN FILE {fname} ---")
            parts.append(contents)
            parts.append(f"--- END FILE {fname} ---")
        return "\n".join(parts).strip()

    def warn_if_large(self, full_prompt, threshold_mb=1.0):
        size_bytes = len(full_prompt.encode("utf-8"))
        size_mb = size_bytes / (1024 * 1024)

        if size_mb > threshold_mb:
            print(f"Warning: prompt size is {size_mb:.2f} MB.")
            print("This may cost dollars rather than cents.")
            resp = input("Continue? [y/N]: ").strip().lower()
            return resp in ("y", "yes")
        return True

    def is_binary_file(self, filename, blocksize=4096):
        try:
            with open(filename, "rb") as f:
                chunk = f.read(blocksize)
        except OSError:
            return False  # treat unreadable as non-binary for now

        # Heuristic: null bytes or too many non-text characters
        if b"\x00" in chunk:
            return True

        # Count non-printable bytes
        text_chars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x7F)))
        nontext = sum(b not in text_chars for b in chunk)

        return nontext / max(1, len(chunk)) > 0.30


    def obfuscate_key(self, api_key):
        import hashlib
        if not api_key:
            return "none"
        h = hashlib.sha256(api_key.encode()).hexdigest()
        return h[:10]   # short, non‑reversible fingerprint


    def log_query_usage(self, filename, model, api_key, upload_bytes, download_bytes):
        import time
        username = os.getlogin()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        key_id = self.obfuscate_key(api_key)
        line = f"{ts}\t{username}\t{model}\t{key_id}\t{upload_bytes}\t{download_bytes}\n"
        with open( filename, "a") as f:
            f.write(line)
        print("Oy", filename, line )


    def log_user_totals(self, filename, model, api_key, upload_bytes, download_bytes):
        import os
        username = os.getlogin()
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
        if username not in totals:
            totals[username] = {
                "uploads": "0",
                "downloads": "0",
                "model": model,
                "key": key_id,
            }

        # Update this user's totals
        totals[username]["uploads"]   = str(int(totals[username]["uploads"].replace(",",""))   + upload_bytes)
        totals[username]["downloads"] = str(int(totals[username]["downloads"].replace(",","")) + download_bytes)
        totals[username]["model"] = model
        totals[username]["key"] = key_id

        # Compute total bytes across all users
        total_bytes_all = 0
        for user, data in totals.items():
            total_bytes_all += int(data["uploads"]) + int(data["downloads"])

        # Compute percentage for each user
        # Store it temporarily in the dict for sorting and writing
        for user, data in totals.items():
            user_bytes = int(data["uploads"]) + int(data["downloads"])
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
                uploads = int(data['uploads'])
                downloads = int(data['downloads'])

                line = ( 
                    f"{user} " 
                    f"pct={pct_str} " 
                    f"uploads={uploads:,} " 
                    f"downloads={downloads:,} " 
                    f"model={data['model']} " 
                    f"key={data['key']}\n"
                )
                f.write(line)

    def user_in_usage_totals(self, username, log_path):
        """
        Returns True if the user already appears in usage_totals.log.
        Otherwise returns False.
        """
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if line.strip().startswith(username + " "):
                        return True
        except FileNotFoundError:
            # No log yet → no users recorded
            return False

        return False

    def require_user_agreement(self, username, ash_dir):
        """
        If the user is new, display site_restrictions.txt (if present)
        and ask for confirmation. Returns True if the user agrees,
        False if they decline.
        """

        restrictions_path = os.path.join(ash_dir, "site_restrictions.txt")

        # If no restrictions file exists, auto-approve
        if not os.path.exists(restrictions_path):
            return True

        # Display restrictions
        print("\n---------------- SITE RESTRICTIONS ----------------")
        try:
            with open(restrictions_path, "r") as f:
                print(f.read().strip())
        except Exception as e:
            print(f"(Warning: could not read site_restrictions.txt: {e})")
        print("---------------------------------------------------\n")

        # Ask for confirmation
        reply = input("Do you agree and wish to proceed? (yes/no): ").strip().lower()

        if reply in ("yes", "y"):
            return True

        print("Request cancelled. You must agree to the site restrictions to continue.")
        return False


    def ai_output_file(self, prompt):
        """
        Parse an output filename from a natural‑language request.
        Returns the filename as a string, or None if not found.
        "output to file foo.txt"          → "foo.txt"
        "write to bar.vcd"                → "bar.vcd"
        "output to the file results.vcd"  → "results.vcd"
        "write to a file named test.vcd"  → "named"  (and this is why we keep it simple)
        """
        text = prompt.lower()
        TRIGGERS = ("output to", "write to")
        SKIP = {"file", "the", "a"}
        for trig in TRIGGERS:
            if trig in text:
                after = text.split(trig, 1)[1].strip()

                # Normalize punctuation
                tokens = after.replace(",", " ").replace(";", " ").split()

                for token in tokens:
                    if token not in SKIP:
                        return token
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
                    if os.path.exists(nxt):
                        # Exclude the output file if present
                        if out_file is None or nxt != out_file:
                            found.append(nxt)
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


# Simple Version
#   def get_env_config(self):
#       api_key = os.environ.get("ASH_API_KEY")
#       model   = os.environ.get("ASH_MODEL", "gemini")  # default
#       user_prompt  = os.environ.get("ASH_USER_PROMPT", "").strip() 
#       logdir  = os.environ.get("ASH_DIR", ".")     # default to cwd
#       return api_key, model, user_prompt, logdir


    def get_provider_config(self):
        provider = os.getenv("ASH_PROVIDER", "gemini")

        if provider == "gemini":
            # Prefer encrypted token, fall back to raw API key
            token = os.getenv("ASH_GEMINI_TOKEN")
            api_key = os.getenv("ASH_GEMINI_API_KEY")

            key = None
            if token:
                secret_key = self.load_site_secret_key();
                username, key = self.decrypt_token( secret_key, token )
                if username != os.getlogin():
                    key = None
            if api_key:
                key = api_key
#           if not key:
#               raise RuntimeError("No API key found for Gemini")          
            # Gemini may default to GEMINI_API_KEY

            return {
                "provider": "gemini",
                "key": key,
                "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
                "model": os.getenv("ASH_MODEL", "gemini-2.0-flash"),
            }

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


    def get_env_config(self):
        """
        Load configuration in this priority order:
            1. User environment variables
            2. Site defaults from $ASH_DIR/site_defaults.txt
            3. Hardcoded internal defaults
        """

        # -----------------------------
        # 1. Load site defaults
        # -----------------------------
        site_defaults = {}
        ash_dir = os.environ.get("ASH_DIR")

        if ash_dir:
            defaults_path = os.path.join(ash_dir, "site_defaults.txt")
            if os.path.exists(defaults_path):
                try:
                    with open(defaults_path, "r") as f:
                        for line in f:
                            line = line.strip()

                            # Skip empty lines and comments
                            if not line or line.startswith("#"):
                                continue

                            # Accept KEY=value or KEY="value"
                            if "=" in line:
                                key, val = line.split("=", 1)
                                key = key.strip()
                                val = val.strip()

                                # Strip optional surrounding quotes
                                if val.startswith('"') and val.endswith('"'):
                                    val = val[1:-1]

                                site_defaults[key] = val
                except Exception as e:
                    print(f"Warning: could not read site_defaults.txt: {e}")

        # -----------------------------
        # 2. Resolve values with priority:
        #    user env > site defaults > internal defaults
        # -----------------------------

        def resolve(key, internal_default):
            # User env wins
            if key in os.environ:
                return os.environ[key]

            # Otherwise site default
            if key in site_defaults:
                return site_defaults[key]

            # Otherwise internal default
            return internal_default

        # Internal defaults (lowest priority)
#       default_model = "gemini-2.5-flash-lite"
        default_user_prompt = ""
        default_logdir = ash_dir if ash_dir else "."

        # Resolve final values
#       api_key     = resolve("ASH_API_KEY", None)
#       provider    = resolve("ASH_PROVIDER", default_model)
#       model       = resolve("ASH_MODEL", default_model)
        user_prompt = resolve("ASH_USER_PROMPT", default_user_prompt)
        logdir      = resolve("ASH_LOG_DIR", default_logdir)

#       return api_key, model, user_prompt, logdir
        return user_prompt, logdir

# ---------- Version ----------
def print_version():
    import os
    import textwrap

    ash_dir = os.environ.get("ASH_DIR", "<not set>")
    model = os.environ.get("ASH_MODEL", "gemini-2.0-flash")

    version_text = f"""
    Ash — AI‑Enabled EDA Assistant
    Version: {ASH_VERSION}

    Ash is a command‑line tool for natural‑language analysis of EDA files.
    It operates using a site‑assigned API key and a configurable AI model.

    Current Configuration
      ASH_DIR:   {ash_dir}
      ASH_MODEL: {model}

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
def print_help( ):
    import os
    import textwrap

    ash_dir = os.environ.get("ASH_DIR", "<not set>")
    model = os.environ.get("ASH_MODEL", "gemini-2.0-flash")  # or whatever default you use

    # Optional site files
    billing_path = os.path.join(ash_dir, "site_billing.txt") if ash_dir != "<not set>" else None
    restrictions_path = os.path.join(ash_dir, "site_restrictions.txt") if ash_dir != "<not set>" else None

    billing_text = ""
    restrictions_text = ""

    if billing_path and os.path.exists(billing_path):
        with open(billing_path, "r") as f:
            billing_text = f.read().strip()

    if restrictions_path and os.path.exists(restrictions_path):
        with open(restrictions_path, "r") as f:
            restrictions_text = f.read().strip()

    help_text = f"""
    Ash — AI‑Enabled EDA Assistant
    Version: {ASH_VERSION}
    Usage: ash [OPTIONS] [COMMAND]

    Ash is a command‑line tool for analyzing Electronic Design Automation (EDA) files
    using natural‑language instructions. It accepts plain English requests such as
    "analyze foo.v" or "summarize timing issues in bar.sdc" and produces structured,
    deterministic output suitable for engineering workflows.

    Ash operates entirely at user level. It does not require elevated privileges and
    does not modify system configuration. All AI requests are performed using the
    API key assigned to this site installation.

    Environment Variables
      ASH_DIR
          Directory containing site configuration files such as:
            site_key.txt          Site secret key (required)
            site_billing.txt      Optional billing information
            site_restrictions.txt Optional usage restrictions
            site_model.txt        Optional default model override

      ASH_PROVIDER
          AI Provider. Defaults to gemini.

      ASH_GEMINI_TOKEN
          Base64‑encoded, per‑user encrypted API key blob. Issued by the site
          administrator. Required for all AI operations to Google Gemini.

      ASH_MODEL
          Overrides the default AI model for this session.
          Current model: {model}

    Changing the Model from the Prompt
      You may request a different model inline:
          "use model gemini‑2.0‑flash and analyze foo.v"
      or by setting the environment variable:
          export ASH_MODEL=gemini‑2.0‑flash
    
    Specifying Input and Output Files in English
     Ash accepts natural‑language file directives:
          "file foo.v"
          "file constraints.sdc"
          "output to file results.txt"

      Multiple input files may be provided:
          "file top.v, file alu.v, analyze timing"

      Output files are written exactly as specified.

    One‑Shot Mode
      When invoked with arguments, Ash runs a single request and exits:
          ash analyze foo.v
          ash summarize timing issues in top.sdc

    License
      Ash (eda_ai_assist) is free software released under the
      GNU General Public License (GPL), version 3 or later.

    Authors
      Kevin Hubbard — Black Mesa Labs
      Additional engineering assistance provided by Microsoft Copilot.

    Site‑Specific Information
    """

    print(textwrap.dedent(help_text).rstrip())

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
def format_prompt() -> str:
    cwd = truncate_string(os.getcwd())
#   return f"{PROMPT_COLOR}{cwd}{RESET_COLOR} % "
    return f"[ash]:{cwd}% "

def truncate_string(input_string, max_length=20):
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

# ---------- Main REPL ----------
def main():
    ai = api_eda_ai_assist()
    shell_type, shell_path = detect_shell()

    _init_readline()
    if on_windows() and not _readline_available:
        print("Note: Tab completion not available. Install `pyreadline3` (pip install pyreadline3) for Windows completion.")

    # Let child processes receive SIGINT; keep wrapper alive
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        return

    if "--version" in sys.argv or "-v" in sys.argv:
        print_version()
        return


    # If arguments were passed, run in "one-shot" mode 
    if len(sys.argv) > 1: # Example: python ash.py analyze foo.txt 
        line = " ".join(sys.argv[1:]) 
        api_key = None
        rts = ai.ask_ai( line, ai_engine="not_gemini", api_key=api_key )
        print( rts )
        return

#   banner = f"{shell_type} wrapper (using {shell_path}). Type `history`, `exit`, or press Ctrl+D to quit."
#   print(banner)
#                                                                                         #  
    print("------------------------------------------------------------------------")
    print("Hi, I'm Ash (eda_ai_assist), your AI EDA assistant from Black Mesa Labs.")
    print("I interpret plain‑English instructions and analyze EDA files.")
    print("Type exit or press Ctrl+D to terminate.")
    print("------------------------------------------------------------------------")



    while True:
        try:
            try:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
            except Exception:
                pass

            line = input(format_prompt())
            if not line.strip():
                continue
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
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
        if stripped in {"exit", "quit"}:
            break

        # Built-in 'history' (show wrapper's numbering)
        if stripped == "history":
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
        if stripped.startswith("cd"):
            parts = stripped.split(maxsplit=1)
            arg = parts[1] if len(parts) == 2 else ""
            handle_cd(arg)
            continue

        if ai.is_ai_request( line ):
#           print( "This was an AI request" )
            api_key = None
            rts = ai.ask_ai( line, ai_engine="not_gemini", api_key=api_key )
            print( rts )
            continue


        # Delegate to the detected shell
        _ = run_shell_command(shell_type, shell_path, line)

    _save_history()
    print("Bye.")

if __name__ == "__main__":
    main()
