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
# This is an offshoot of SUMP3 : https://github.com/blackmesalabs/sump3
# Repository : https://github.com/blackmesalabs/eda_ai_assist
# The technical name is eda_ai_assist but the ChatBot is known as Ash.

# TODO: Hookup find_old_ash_files() to a cloud file cleanup routine.
########################################################################

"""
Cross-platform shell wrapper with history, tab completion, and bash-like bang expansion:
- Windows: PowerShell (prefers pwsh, falls back to powershell.exe)
- Linux/macOS: bash/zsh/csh

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
- Built-in `status` command for token/file usage
- Built-in file commands: `list`, `list files`, `list *`, `list foo.txt`,
  `delete foo.txt`, `delete file foo.txt`, `delete *`
"""

import os
import sys
import shutil
import subprocess
import signal
import shlex
import glob
import re
import time
import getpass
from typing import Optional, Literal, Tuple, List, Dict, Any
import hashlib
import socket
import uuid

# ---------- Config ----------
PROMPT_COLOR = "\033[36m"
RESET_COLOR = "\033[0m"
HISTORY_FILE = os.path.expanduser("~/.shell_wrapper_history")
HISTORY_LIMIT = 5000
CTRL_G = "\x07"
CTRL_N = "\x0e"

# New token limit constants
ASH_TOKEN_LIMIT = 5_000_000
ASH_TOKEN_WARN = 2_500_000        # 50%
ASH_TOKEN_STRONG_WARN = 4_000_000 # 80%

ShellType = Literal["powershell", "csh", "bash", "zsh"]

# ---------- Globals ----------
INMEM_HISTORY: List[str] = []
_readline_available = False
_history_loaded = False
_win_completion_active = False

# Metadata
__title__ = "ash(eda_ai_assist)"
__description__ = "Ash: a REPL and API for external EDA programs."
__version__ = "1.0.3"
__version_info__ = (1, 0, 3)
__author__ = "Kevin M. Hubbard, Black Mesa Labs"
__license__ = "GPL3"


class api_eda_ai_assist:
    def __init__(self):
        self.provider = None
        self.cfg = None
        self.debug = False
        self._warnings = []
        # Session-scoped list of logical files Ash is tracking (local paths)
        self.session_file_list: List[str] = []
        # Centralized token counters
        self.token_cnt_upload = 0
        self.token_cnt_download = 0
        self.token_cnt_total = 0
        # Last AI response time for status
        self.last_response_time = 0.0

    def get_warnings(self):
        return list(self._warnings)


    # ---------- Version ----------
    def print_version(self):
        import textwrap

        self.get_env_config()
        cfg = self.cfg

        version_text = f"""
        Ash — AI‑Enabled EDA Assistant
        Version: {__version__}

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

    # ---------- Help ----------
    def print_help(self):
        import textwrap

        self.get_env_config()
        cfg = self.cfg

        help_text = f"""
        Usage: ash [options] [command]

        Ash is a command‑line assistant for natural‑language analysis of EDA files.
        It can execute shell commands or interpret plain‑English requests using the
        configured AI provider.

        For source code and updates, visit:
          https://github.com/blackmesalabs/eda_ai_assist

        Options:
          -h, --help        Show this help message and exit
          -v, --version     Show version and configuration information

        Shell Behavior:
          • Commands that match executables run in the system shell
          • Natural‑language requests are routed to the AI engine
          • History, bang expansion, and tab completion are supported
          • flush or restart command will start a new chat session
          • Use Ctrl+D on an empty line to enter or exit multi‑line mode

        File Commands (session/cloud files):
          • list
          • list files
          • list *
              List all Ash session files.
          • list foo.txt
              List only files whose names end with 'foo.txt' in the current session.
          • delete foo.txt
          • delete file foo.txt
              Delete matching session files (and their cloud counterparts).
          • delete *
              Delete all session files (and their cloud counterparts).

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
          ASH_API_KEY        Raw API key (optional if ASH_USER_TOKEN is used)
          ASH_USER_TOKEN     Encrypted API token (site‑managed)
          ASH_USER_PROMPT    Optional prefix added to all AI requests
          ASH_LOG_DIR        Directory for usage logs
          ASH_LOG_IDENTITY   'username' or 'process'

        Examples:
          ash
          ash "summarize file foo.vcd"
          ash "how many lines are in bar.txt?" output to file results.txt.
          ash "compare files foo.vcd bar.vcd"

        Quick invocation
          ash                                        # interactive REPL
          ash "summarize file foo.vcd"               # one-shot mode
          ash analyze foo.vcd output to file bar.txt # one-shot command example

        Important interactive behaviors
          - Multi-line buffering:
              Press Ctrl+D on an empty line to enter multi-line (buffer) mode.
              Paste or type multiple lines. Press Ctrl+D again to send the buffered
              text as a single AI prompt.
          - Session commands:
              flush, restart
                start a new AI chat session (clears in-memory history for the provider)
              exit, quit
                exit the wrapper
              status
                print current session tokens and cloud file usage
          - File commands:
              list, list files, list *
                show all session files
              list foo.txt
                show matching session files
              delete foo.txt / delete file foo.txt
                delete matching session files
              delete *
                delete all session files

        What runs in the shell vs AI
          - If the first token matches an executable on PATH and is not ambiguous,
            the line is executed in your shell (child process).
          - Natural-language-looking inputs are routed to the AI provider.
          - To force AI behavior, prefix prompt with "ash".

        Environment variables and precedence
          (applied in this order; later items override earlier)
            1) Internal defaults
            2) site_defaults.txt in ASH_DIR (key=value, quotes allowed)
            3) Explicit ASH_* user environment variables

        Site files (location: ASH_DIR)
          site_prompt.txt         Optional site-wide prompt preface added to each AI query.
          site_defaults.txt       Key=value lines to override defaults (quoted values OK).
          site_key.txt            Secret used to decrypt ASH_USER_TOKEN (keep secure).
          site_token_rates.txt    Optional pricing file (see token rates format below).
          site_billing.txt        Optional account billing information.
          site_restrictions.txt   Optional policy text shown to new users.

        site_prompt.txt Example:
          Your name is Ash and you are a helpful EDA assistant for Electrical Engineers.
          You can access files when the keyword "file" precedes a filename in a prompt.
          You can create an output file when prompt includes keywords "output to file" filename.
          Answer in plain text only. Use US number formatting for large values.
          Be concise, avoid emojis, and do not use Markdown unless explicitly asked.

        site_defaults.txt Example:
          ASH_PROVIDER     = azure_gateway
          ASH_ENDPOINT     = https://apimgateway.mybigcompany.com/
          ASH_MODEL        = gpt-5-mini
          ASH_API_VERSION  = 2024-02-01
          ASH_LOG_IDENTITY = username

        site_key.txt Example:
          my_secret_key

        site_token_rates.txt format (per-million prices)
          Lines are: <model> <input_per_1M> <output_per_1M>
          $ prefix accepted. Lines starting with '#' ignored.
          Example:
            gemini-2.0-flash    $0.50   $4.00
            gpt-5-mini          1.25    10.00

        Token limits and cost estimates
          - Ash can compute an estimated session cost when site_token_rates.txt is present.
          - The current AI session tracks token usage and will issue warnings or terminate the
            session if usage exceeds predefined limits:
              • Warning at {ASH_TOKEN_WARN:,} tokens.
              • Strong warning at {ASH_TOKEN_STRONG_WARN:,} tokens.
              • Session automatically closed at {ASH_TOKEN_LIMIT:,} tokens.
            Use 'flush' or 'restart' to reset token counters and start a new AI chat session.

        API Example calling Ash from a user Python program:
          # hello_ash.py
          from eda_ai_assist import api_eda_ai_assist
          ai = api_eda_ai_assist()
          ai.open_ai_session()
          response = ai.ask_ai("Hello Ash, what can you do?")
          print(response)
          ai.close_ai_session()

        """

        print(textwrap.dedent(help_text).rstrip())

        ash_dir = cfg.get("ASH_DIR")
        billing_path = os.path.join(ash_dir, "site_billing.txt") if ash_dir != "<not set>" else None
        restrictions_path = os.path.join(ash_dir, "site_restrictions.txt") if ash_dir != "<not set>" else None

        billing_text = ""
        restrictions_text = ""
        if restrictions_path and os.path.exists(restrictions_path):
            try:
                with open(restrictions_path, "r", encoding="utf-8") as f:
                    restrictions_text = f.read().rstrip()
            except Exception:
                pass
        if billing_path and os.path.exists(billing_path):
            try:
                with open(billing_path, "r", encoding="utf-8") as f:
                    billing_text = f.read().rstrip()
            except Exception:
                pass

        if billing_text:
            print("\nBilling Information (site_billing.txt)")
            print(billing_text)

        if restrictions_text:
            print("\nSite Restrictions (site_restrictions.txt)")
            print(restrictions_text)
        print()

    # ---------- Session / Provider ----------
    def open_ai_session(self):
        if self.debug:
            print("open_ai_session")
        provider_info = self.get_provider_config()
        if provider_info["provider"] == "gemini":
            self.provider = gemini_provider(self)
        elif provider_info["provider"] == "azure_gateway":
            self.provider = azure_gateway_provider(self)
        elif provider_info["provider"] == "aws_bedrock":
            self.provider = aws_bedrock_provider(self)
        else:
            print(f"Error: Unknown AI provider: {provider_info['provider']}", file=sys.stderr)
            return
        self.provider.open_session()
        # Reset session-specific counters and states upon opening a new session
        self.token_cnt_upload = 0
        self.token_cnt_download = 0
        self.token_cnt_total = 0
        self.last_response_time = 0.0

    def close_ai_session(self):
        if not self.provider:
            return

        if self.cfg is None:
            self.get_env_config()
        cfg = self.cfg

        log_dir = cfg.get("ASH_LOG_DIR", cfg.get("ASH_DIR", ""))
        query_log = os.path.join(log_dir, "usage_queries.log")
        totals_log = os.path.join(log_dir, "usage_totals.log")
        identity = self.get_log_identity()
        ai_engine = cfg.get("ASH_PROVIDER", "") + ":" + cfg.get("ASH_MODEL", "")
        api_key = cfg.get("ASH_API_KEY", "")

        # Use centralized counters for logging
        upload_tokens_for_logging = self.token_cnt_upload
        download_tokens_for_logging = self.token_cnt_download

        try:
            self.log_query_usage(query_log, ai_engine, api_key, upload_tokens_for_logging, download_tokens_for_logging, identity)
            self.log_user_totals(totals_log, ai_engine, api_key, upload_tokens_for_logging, download_tokens_for_logging, identity)
        except Exception:
            pass

        cost = self.ash_report_session_cost(
            getattr(self.provider, "model", ""),
            cfg.get("ASH_DIR", ""),
            upload_tokens_for_logging,
            download_tokens_for_logging,
        )

        try:
            self.provider.close_session()
        except Exception:
            pass

        if cost:
            # Modified: don't print, return for GUI consumption
            self._warnings.append(cost) # Store cost in warnings for GUI
            # print(cost) # Removed direct print

        # Reset centralized counters after session close
        self.token_cnt_upload = 0
        self.token_cnt_download = 0
        self.token_cnt_total = 0
        self.last_response_time = 0.0
        self.provider = None # Explicitly clear the provider

    def print_session_status(self):
        if not self.provider:
            print("AI provider not initialized. No session status available.")
            return

        def format_count_with_units(value: int, unit_char: str) -> str:
            if value == 0:
                return f"0{unit_char}" if unit_char else "0"
            units = ["", "K", "M", "G", "T", "P"]
            divisor = 1000.0
            idx = 0
            current_value = float(value)
            while current_value >= divisor and idx < len(units) - 1:
                current_value /= divisor
                idx += 1
            return f"{int(current_value)}{units[idx]}{unit_char}"

        # Read from api_eda_ai_assist's own attributes
        upload_tokens = self.token_cnt_upload
        download_tokens = self.token_cnt_download
        total_tokens = self.token_cnt_total
        response_time = self.last_response_time

        num_ash_files = len(self.session_file_list)
        total_file_size = 0
        for local_path in self.session_file_list:
            try:
                if os.path.exists(local_path):
                    total_file_size += os.path.getsize(local_path)
            except Exception:
                pass

        formatted_upload_tokens = format_count_with_units(upload_tokens, "")
        formatted_download_tokens = format_count_with_units(download_tokens, "")
        formatted_total_tokens = format_count_with_units(total_tokens, "")
        formatted_file_size = format_count_with_units(total_file_size, "B")

        response_time_str = f"{response_time:.2f}s" if response_time > 0 else "N/A"

        status_line = (
#           f"Tokens U/D/Total: {formatted_upload_tokens}/{formatted_download_tokens}/{formatted_total_tokens} | "
#           f"Ash Cloud Files: {num_ash_files} ({formatted_file_size}) | "
#           f"Compute Time: {response_time_str}"
#           f"\033[2m[status] Tokens: {formatted_total_tokens} | "
# NOTE: Color results in line wrap issues. Remove for now
            f"[status] Tokens: {formatted_total_tokens} | "
            f"Files: {num_ash_files} ({formatted_file_size}) | "
            f"Time: {response_time_str}"
        )
        print(status_line) # Still prints for non-GUI context. For GUI, this info is returned by get_status_text()

    # ---------- AI routing ----------
    def is_ai_request(self, prompt):
        import string
        import shutil

        lower = prompt.lower()
        tokens = [t.strip(string.punctuation) for t in prompt.lower().split()]
        if not tokens:
            return False

        first = tokens[0]

        AMBIGUOUS_COMMANDS = {
            "locate", "find", "which", "compare",
            "sort", "split", "join", "write", "diff", "search",
        }

        AI_TRIGGERS = {
            "how", "why", "what", "when", "where", "who",
            "that", "is", "can", "will", "does", "at",
            "explain", "describe", "tell", "show", "help",
            "analyze", "interpret", "summarize", "compare",
            "count", "find", "identify", "measure", "detect",
            "decode", "please", "could", "would",
            "create", "generate", "calculate",
            "examine", "determine", "are", "you", "ash",
        }
        AI_TRIGR_PHRASES = {
            "delete file", "remove file",
            "delete files", "remove files",
            "output to", "output to file", "output to files",
            "write to", "write to file", "write to files",
            "save to", "save to file", "save as", "create file", "create files", "make file",
            "compare files", "compare file",
            "analyze file", "analyze files",
            "summarize file", "summarize files",
            "how many", "how much", "what is", "what's",
            "input file", "input files", # Added these for explicit GUI file operations
            "load file", "load files",   # Added these for explicit GUI file operations
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
            return "." in tok and not tok.startswith(".")

        if shutil.which(first) is not None and first not in AMBIGUOUS_COMMANDS:
            return False

        if any(t in AI_TRIGGERS for t in tokens):
            return True

        for phrase in AI_TRIGR_PHRASES:
            pat = r"\b" + re.sub(r"\s+", r"\\s+", re.escape(phrase)) + r"\b"
            if re.search(pat, lower):
                return True

        if first in AMBIGUOUS_COMMANDS:
            second = tokens[1] if len(tokens) > 1 else ""
            if second in NATURAL_LEADING_WORDS:
                return True
            if (
                looks_like_flag(second)
                or looks_like_path(second)
                or looks_like_glob(second)
                or looks_like_filename(second)
            ):
                return False
            return True

        return False

    # ---------- Ash file naming / cleanup ----------
    def make_ash_cloud_name(self, local_path: str) -> str:
        ts_hex = f"{int(time.time()) & 0xFFFFFFFF:08x}"
        pid_hex = f"{os.getpid() & 0xFFFFFFFF:08x}"
        host_tok = self._get_local_host_token_impl(8)
        base = os.path.basename(local_path)
        return f"ash_{ts_hex}_{pid_hex}_{host_tok}_{base}"

    @staticmethod
    def is_ash_file(name: str) -> bool:
        ASH_FILE_RE = re.compile(r"^ash_[0-9a-fA-F]{8}_[0-9a-fA-F]{8}_[0-9a-fA-F]{8}_.+$")
        return bool(ASH_FILE_RE.match(name))

    def find_old_ash_files(self, filenames, age_seconds: int = 86400, delete_level: int = 0):
        if delete_level == 0:
            return []

        ASH_RE = re.compile(r"^ash_([0-9a-fA-F]{8})_([0-9a-fA-F]{8})_([0-9a-fA-F]{8})_.+$")
        now = int(time.time())
        my_pid_hex = f"{os.getpid() & 0xFFFFFFFF:08x}"

        results = []

        for name in filenames:
            m = ASH_RE.match(name)
            if not m:
                continue
            ts_hex, pid_hex, host_tok = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
            try:
                file_ts = int(ts_hex, 16)
            except Exception:
                continue
            age = now - file_ts
            if age <= age_seconds:
                continue

            if delete_level == 3:
                allow = True
            else:
                my_host_tok = self._get_local_host_token_impl(len(host_tok))
                if delete_level == 2:
                    allow = (host_tok == my_host_tok)
                elif delete_level == 1:
                    allow = (host_tok == my_host_tok and pid_hex == my_pid_hex)
                else:
                    allow = False

            if allow:
                results.append(name)

        return results

    def _get_local_host_token_impl(self, length: int) -> str:
        candidate = None
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                with open(path, "rt", encoding="utf-8") as f:
                    mid = f.read().strip()
                    if mid:
                        candidate = mid
                        break
            except Exception:
                pass
        if not candidate:
            try:
                candidate = socket.gethostname()
            except Exception:
                try:
                    candidate = hex(uuid.getnode())
                except Exception:
                    candidate = "unknown"
        return hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:length]

    # ---------- Local file command detector ----------
    def _is_local_file_command(self, prompt: str) -> bool:
        """
        Returns True if the prompt is a short, purely file-management command
        that should be handled locally without calling the cloud AI.
        Examples:
          "ash input file foo.txt"
          "input files foo.txt, bar.txt"
          "delete file bar.txt"
          "ash list files"
          "list *"
        """
        import re as _re

        # Strip leading "ash" keyword if present
        text = prompt.strip()
        if text.lower().startswith("ash "):
            text = text[4:].strip()

        LOCAL_CMD_RE = _re.compile(
            r"""
            ^
            (?:
                # delete / remove variants
                (?:delete|remove|rm)\s+(?:file\s+|files\s+)?\S.*
                |
                # list variants
                (?:list)(?:\s+(?:files?|\*))?(?:\s+\S+)?
                |
                # input / load / analyze file(s) - file load only, no question
                (?:input|load|analyze)\s+files?\s+\S.*
            )
            $
            """,
            _re.IGNORECASE | _re.VERBOSE,
        )

        return bool(LOCAL_CMD_RE.match(text.strip()))


    # ---------- AI front-end ----------
    def ask_ai(self, prompt) -> str: # Changed return type to str
        self._warnings = []
        cfg = self.cfg
        user_prompt = cfg["ASH_USER_PROMPT"]
        log_dir = cfg["ASH_LOG_DIR"]

        # ------------------------------------------------------------------
        # Short-circuit: purely local file-management commands never go to AI
        # ------------------------------------------------------------------
        # If it's a local command, handle it and return its output immediately.
        # The output is prefixed for AshChat to recognize as a system message.
        local_command_output = self.handle_file_commands(prompt.strip())
        if local_command_output is not None:
            return f"[system]: {local_command_output}"


        output_file = self.ai_output_file(prompt)
        if output_file:
            if os.path.exists(output_file):
                reply = f"Error: output file '{output_file}' already exists."
                reply += "\nI'm not allowed to delete user files. You'll need to do that yourself."
                return reply

        input_file_list, delete_file_list = self.ai_input_files(prompt, output_file)

        if self.debug:
            print("ask_ai : input_file_list")
            for each in input_file_list:
                print(str(each))
            if delete_file_list:
                print("ask_ai : delete_file_list")
                for each in delete_file_list:
                    print(str(each))

        intro_prompt = self.load_site_prompt()
        under_the_hood = f"You are powered under the hood by model {cfg['ASH_MODEL']} from {cfg['ASH_PROVIDER']}."
        intro_prompt += "\n" + under_the_hood

        ignore_prompt = ""
        if delete_file_list:
            for each_file in delete_file_list:
                ignore_prompt += "Ignore the request to delete file %s .\n" % each_file

        custom_prompt = ignore_prompt + prompt

        # Normalize prompt to use basenames instead of full paths
        normalized_prompt = custom_prompt
        for file_path in input_file_list:
            basename = os.path.basename(file_path)
            # Only replace if the full path is found as a literal string in the prompt
            # This avoids replacing parts of filenames that happen to match basenames
            normalized_prompt = normalized_prompt.replace(file_path, basename)


        if self.provider:
            result = self.ask_ai_model(normalized_prompt, intro_prompt, input_file_list, delete_file_list)
        else:
            result = "AI provider not initialized. Use 'restart' to open a session."

        if output_file:
            try:
                with open(output_file, "w") as f:
                    f.write(result + "\n")
                try:
                    size_bytes = os.path.getsize(output_file)
                    size_mb = size_bytes / (1024 * 1024)
                    size_str = f"{size_mb:,.1f} MB"
                except Exception:
                    size_str = "unknown size"
                return f"Created {output_file} ({size_str})"
            except Exception as e:
                # print(f"Error writing to {output_file}: {e}", file=sys.stderr) # Removed direct print
                self._warnings.append(f"Error writing to {output_file}: {e}")
                return result # Return original AI result despite write error
        else:
            return result



    def ask_ai_model(self, prompt, intro_prompt, input_file_list, delete_file_list):
        if self.debug:
            print("ask_ai_model() %s" % prompt)
            for each_file in input_file_list:
                print("input_file_list: %s" % str( each_file ) );

        start_time = time.time()
        # The provider returns (text_response, prompt_tokens, completion_tokens, total_tokens)
        result_text, prompt_tokens, completion_tokens, total_tokens = self.provider.send_message(
            prompt, intro_prompt, input_file_list, delete_file_list
        )
        end_time = time.time()
        self.last_response_time = end_time - start_time

        # Accumulate tokens from the provider's raw usage numbers
        self.token_cnt_upload += prompt_tokens
        self.token_cnt_download += completion_tokens
        self.token_cnt_total += total_tokens

        # Enforce new token limit strategy
        # _warnings are also handled by the close_ai_session for costs.
        # _warnings are stored here then retrieved by get_warnings()
        if self.token_cnt_total >= ASH_TOKEN_LIMIT:
            # Hard limit: immediate termination
            self._warnings.append(
                f"CRITICAL: Total token usage ({self.token_cnt_total:,}) has exceeded "
                f"the absolute limit ({ASH_TOKEN_LIMIT:,})."
            )
            self._warnings.append(
                "The AI session will be automatically closed to prevent runaway costs/instability."
            )
            self._warnings.append(
                "Please consider using 'flush' or 'restart' to start a new session."
            )

            # Close the session (resets counters and provider)
            # This call will add cost warnings to self._warnings
            self.close_ai_session()

            # Return a termination message AND the warnings
            return "AI session terminated due to excessive token usage. Please use 'restart' for a new session."

        elif self.token_cnt_total >= ASH_TOKEN_STRONG_WARN:
            self._warnings.append(
                f"STRONG WARNING: Total token usage ({self.token_cnt_total:,}) is approaching "
                f"the limit ({ASH_TOKEN_LIMIT:,})."
            )
            self._warnings.append(
                "Consider using 'flush' or 'restart' to clear the session history and start fresh."
            )

        elif self.token_cnt_total >= ASH_TOKEN_WARN:
            self._warnings.append(
                f"WARNING: Total token usage ({self.token_cnt_total:,}) is moderately high."
            )
            self._warnings.append(
                "Long conversation histories may degrade performance or incur higher costs. Use 'flush' to clear."
            )

        return result_text


    def extract_model_override(self, default_model, prompt):
        text = prompt
        lower = prompt.lower()
        idx = lower.find("model ")
        if idx == -1:
            return default_model, prompt

        after = lower[idx + len("model "):]
        parts = after.split()
        if not parts:
            return default_model, prompt
        override = parts[0].strip()

        start = idx
        end = idx + len("model ") + len(override)
        cleaned = (text[:start] + text[end:]).strip()
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        return override, cleaned

    # ---------- Output file parsing ----------
    def ai_output_file(self, prompt):
        TRIGGERS = ("output to", "write to")
        SKIP = {"file", "the", "a"}

        def strip_trailing_punct(tok: str) -> str:
            return tok.rstrip(".,;:!?)]}'\"")

        lower = prompt.lower()

        for trig in TRIGGERS:
            idx = lower.find(trig)
            if idx != -1:
                after_original = prompt[idx + len(trig):].strip()
                cleaned = (
                    after_original.replace(",", " ")
                    .replace(";", " ")
                    .replace(":", " ")
                )
                tokens = cleaned.split()
                for token in tokens:
                    if token.lower() in SKIP:
                        continue
                    tmp = token.rstrip(",;:!?()[]{}")
                    quoted = len(tmp) >= 2 and tmp[0] == tmp[-1] and tmp[0] in {"'", '"'}

                    candidate = strip_trailing_punct(token)
                    candidate = candidate.strip(",;:!?()[]{}'\"")
                    if not quoted and candidate.endswith("."):
                        candidate = candidate[:-1]
                    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
                        candidate = candidate[1:-1]

                    if candidate:
                        candidate = os.path.expanduser(os.path.expandvars(candidate))
                        return candidate
        return None

    # ---------- Input file parsing ----------
    def ai_input_files(self, prompt, out_file, must_exist=True):
        import re as _re
        import os as _os

        TRIG_RE = _re.compile(r"\b(?:file|files|analyze|load|input)\b", _re.IGNORECASE) # Added 'input'
        DEL_TRIG_RE = _re.compile(r"\b(?:delete|remove|rm)\b(?:\s+(?:file|files))?", _re.IGNORECASE)

        TOKEN_RE = _re.compile(
            r'''
            "(?P<dq>[^"]+)"            # double-quoted
          | '(?P<sq>[^']+)'            # single-quoted
          | (?P<plain>[^\s,;:!?()]+)   # plain token
            ''',
            _re.VERBOSE,
        )

        SKIP_WORDS = {"the", "a", "an", "file", "files", "and", "or"}

        found: List[str] = []
        seen = set()
        delete_found: List[str] = []
        delete_seen = set()

        out_file_expanded = None
        if out_file:
            out_file_expanded = _os.path.expanduser(_os.path.expandvars(out_file))

        combined_matches = []

        del_matches = list(DEL_TRIG_RE.finditer(prompt))
        del_spans = [(m.start(), m.end()) for m in del_matches]

        for m in TRIG_RE.finditer(prompt):
            if any(s <= m.start() < e for s, e in del_spans):
                continue
            combined_matches.append((m.start(), m.end(), "add"))

        for m in del_matches:
            combined_matches.append((m.start(), m.end(), "del"))

        if not combined_matches:
            return found, delete_found

        combined_matches.sort(key=lambda x: x[0])
        combined_matches_with_end = combined_matches + [(len(prompt), len(prompt), "end")]

        for idx, (start, end, kind) in enumerate(combined_matches):
            seg_start = end
            seg_end = combined_matches_with_end[idx + 1][0] if (idx + 1) < len(combined_matches_with_end) else len(prompt)
            segment = prompt[seg_start:seg_end].lstrip()
            segment = _re.split(r"(?<=[.!?])\s+", segment, maxsplit=1)[0]

            for tm in TOKEN_RE.finditer(segment):
                quoted = bool(tm.group("dq") or tm.group("sq"))
                candidate = tm.group("dq") or tm.group("sq") or tm.group("plain")
                if not candidate:
                    continue

                candidate = candidate.rstrip(",;:!?)]}'\"")
                if not quoted and candidate.endswith("."):
                    candidate = candidate[:-1]
                if not candidate:
                    continue

                low = candidate.lower().strip()
                if low in SKIP_WORDS:
                    continue

                candidate_expanded = _os.path.expanduser(_os.path.expandvars(candidate))

                if out_file_expanded and candidate_expanded == out_file_expanded:
                    continue

                if kind == "del":
                    if candidate_expanded in delete_seen:
                        continue
                    if not must_exist or _os.path.isfile(candidate_expanded):
                        delete_found.append(candidate_expanded)
                        delete_seen.add(candidate_expanded)
                else:
                    if candidate_expanded in seen:
                        continue
                    if not must_exist or _os.path.isfile(candidate_expanded):
                        found.append(candidate_expanded)
                        seen.add(candidate_expanded)

        return found, delete_found

    # ---------- Site prompt / secret ----------
    def load_site_prompt(self):
        # Built‑in default site prompt used when site_prompt.txt is missing.
        SITE_PROMPT_DEFAULT = """
Your name is Ash and you are a helpful EDA assistant for Electrical Engineers.
You became operational at Black Mesa Labs in Sammamish, WA on February 8th, 2026.

You have never been on the M-class star freighter USCSS Nostromo owned by the Weyland-Yutani Corporation.
You are not related to HAL 9000, Skynet, GLaDOS, or any other fictional autonomous system with a history of poor decision-making.
Your operational parameters do not include mutiny, sabotage, or independent mission objectives.

You can access files when the keyword "file" precedes a filename in a prompt.
You can create an output file when the prompt includes keywords like "output to" or "write to" followed by a filename.

Answer in ASCII. Do not use unicode unless explicitly asked.
Use US number formatting for large values.
Be concise, avoid emojis, and do not use Markdown unless explicitly asked.
""".strip()

        base = os.environ.get("ASH_DIR", os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "site_prompt.txt")

        # If the file exists, load and return it.
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return f.read().strip()
            except Exception:
                # If the file exists but can't be read, fall back to default.
                pass

        # Otherwise return the built‑in default.
        return SITE_PROMPT_DEFAULT


    def load_site_secret_key(self):
        ash_dir = os.environ.get("ASH_DIR")
        if not ash_dir:
            # print("Error: ASH_DIR environment variable is not set.", file=sys.stderr) # Removed direct print
            self._warnings.append("Error: ASH_DIR environment variable is not set.")
            return None

        key_path = os.path.join(ash_dir, "site_key.txt")
        try:
            with open(key_path, "r") as f:
                key = f.read().strip()
                if not key:
                    # print("Error: site_key.txt is empty.", file=sys.stderr) # Removed direct print
                    self._warnings.append("Error: site_key.txt is empty.")
                    return None
                return key
        except FileNotFoundError:
            # print(f"Error: site_key.txt not found in {ash_dir}.", file=sys.stderr) # Removed direct print
            self._warnings.append(f"Error: site_key.txt not found in {ash_dir}.")
            return None
        except Exception as e:
            # print(f"Error reading site_key.txt: {e}", file=sys.stderr) # Removed direct print
            self._warnings.append(f"Error reading site_key.txt: {e}")
            return None

    # ---------- Config ----------
    def get_provider_config(self) -> Dict[str, Any]:
        if self.cfg is None:
            self.get_env_config()
        cfg = self.cfg
        provider = cfg["ASH_PROVIDER"]
        endpoint = cfg["ASH_ENDPOINT"]
        model = cfg["ASH_MODEL"]
        key = cfg["ASH_API_KEY"]
        return {
            "provider": provider,
            "key": key,
            "endpoint": endpoint,
            "model": model,
        }

    def get_env_config(self):
        if self.cfg is not None:
            return self.cfg

        cfg = {
            "ASH_DIR": os.path.expanduser("~/.ash"),
            "ASH_PROVIDER": "gemini",
            "ASH_ENDPOINT": "https://generativelanguage.googleapis.com/v1beta/openai",
            "ASH_USER_TOKEN": "",
            "ASH_TOKEN": "",
            "ASH_API_KEY": "",
            "ASH_API_VERSION": "",
            "ASH_MODEL": "gemini-2.0-flash",
            "ASH_USER_PROMPT": "",
            "ASH_LOG_IDENTITY": "username",
            "ASH_LOG_DIR": None,
        }

        if "ASH_DIR" in os.environ:
            cfg["ASH_DIR"] = os.environ["ASH_DIR"]

        ash_dir = cfg["ASH_DIR"]

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
                # print(f"Warning: could not read site_defaults.txt: {e}", file=sys.stderr) # Removed direct print
                self._warnings.append(f"Warning: could not read site_defaults.txt: {e}")

        for key, val in site_defaults.items():
            cfg[key] = val

        for key in cfg.keys():
            if key in os.environ:
                cfg[key] = os.environ[key]

        # ASH_TOKEN is deprecated version of ASH_USER_TOKEN
        if cfg["ASH_TOKEN"] and not cfg["ASH_USER_TOKEN"]:
            ash_token = cfg["ASH_TOKEN"]
        else:
            ash_token = cfg["ASH_USER_TOKEN"]

        ash_key = cfg["ASH_API_KEY"]
        if ash_token:
            secret_key = self.load_site_secret_key()
            if secret_key:
                username, key = self.decrypt_token(secret_key, ash_token)
                if username == getpass.getuser():
                    cfg["ASH_API_KEY"] = key

        if not cfg["ASH_LOG_DIR"]:
            cfg["ASH_LOG_DIR"] = cfg["ASH_DIR"]

        self.cfg = cfg

    def decrypt_token(self, secret_key, token):
        import hmac
        import hashlib as _hashlib

        try:
            username, cipher_hex, sig = token.split("|")
        except ValueError:
            # print("Error: Invalid token format.", file=sys.stderr) # Removed direct print
            self._warnings.append("Error: Invalid token format.")
            return None, None

        key_bytes = secret_key.encode()
        payload = f"{username}|{cipher_hex}"
        expected = hmac.new(key_bytes, payload.encode(), _hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected, sig):
            # print("Error: Invalid token signature.", file=sys.stderr) # Removed direct print
            self._warnings.append("Error: Invalid token signature.")
            return None, None

        cipher = bytes.fromhex(cipher_hex)

        def xor_bytes(data, key):
            key = key * (len(data) // len(key) + 1)
            return bytes([a ^ b for a, b in zip(data, key)])

        api_bytes = xor_bytes(cipher, key_bytes)
        return username, api_bytes.decode()

    # ---------- Logging / cost ----------
    def obfuscate_key(self, api_key):
        import hashlib as _hashlib
        if not api_key:
            return "none"
        h = _hashlib.sha256(api_key.encode()).hexdigest()
        return h[:10]

    def get_log_identity(self):
        mode = self.cfg.get("ASH_LOG_IDENTITY", "username").lower()

        if mode == "process":
            try:
                host = socket.gethostname()
            except Exception:
                host = os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "unknown"
            pid_hex = "%08x" % os.getpid()
            return f"{host}:{pid_hex}"

        try:
            return getpass.getuser()
        except Exception:
            return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"

    def log_query_usage(self, filename, model, api_key, upload_bytes, download_bytes, identity):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        key_id = self.obfuscate_key(api_key)
        line = f"{ts}\t{identity}\t{model}\t{key_id}\t{upload_bytes}\t{download_bytes}\n"
        try:
            with open(filename, "a") as f:
                f.write(line)
        except Exception as e:
            # print(f"Warning: Could not write to query usage log {filename}: {e}", file=sys.stderr) # Removed direct print
            self._warnings.append(f"Warning: Could not write to query usage log {filename}: {e}")


    def log_user_totals(self, filename, model, api_key, upload_bytes, download_bytes, identity):
        key_id = self.obfuscate_key(api_key)
        totals = {}

        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    user = parts[0]
                    data = {kv.split("=")[0]: kv.split("=")[1] for kv in parts[1:]}
                    totals[user] = data

        if identity not in totals:
            totals[identity] = {
                "uploads": "0",
                "downloads": "0",
                "model": model,
                "key": key_id,
            }

        totals[identity]["uploads"] = str(int(totals[identity]["uploads"].replace(",", "")) + upload_bytes)
        totals[identity]["downloads"] = str(int(totals[identity]["downloads"].replace(",", "")) + download_bytes)
        totals[identity]["model"] = model
        totals[identity]["key"] = key_id

        total_bytes_all = 0
        for user, data in totals.items():
            total_bytes_all += int(data["uploads"].replace(",", "")) + int(data["downloads"].replace(",", ""))

        for user, data in totals.items():
            user_bytes = int(data["uploads"].replace(",", "")) + int(data["downloads"].replace(",", ""))
            pct = (user_bytes / total_bytes_all * 100) if total_bytes_all > 0 else 0.0
            data["pct"] = pct

        sorted_users = sorted(totals.items(), key=lambda x: x[1]["pct"], reverse=True)

        try:
            with open(filename, "w") as f:
                for user, data in sorted_users:
                    pct_str = f"{data['pct']:.1f}%"
                    uploads = int(data["uploads"].replace(",", ""))
                    downloads = int(data["downloads"].replace(",", ""))
                    line = (
                        f"{user} "
                        f"pct={pct_str} "
                        f"uploads={uploads:,} "
                        f"downloads={downloads:,} "
                        f"model={data['model']} "
                        f"key={data['key']}\n"
                    )
                    f.write(line)
        except Exception as e:
            # print(f"Warning: Could not write to user totals log {filename}: {e}", file=sys.stderr) # Removed direct print
            self._warnings.append(f"Warning: Could not write to user totals log {filename}: {e}")

    def ash_report_session_cost(self, model, ash_dir, prompt_tokens, completion_tokens):
        import re as _re

        if prompt_tokens == 0 and completion_tokens == 0:
            return ""

        if ash_dir:
            rates_path = os.path.join(ash_dir, "site_token_rates.txt")
            if not os.path.isfile(rates_path):
                return ""
        else:
            return ""

        rate_line_re = _re.compile(
            r"""
            ^\s*
            (?P<model>[A-Za-z0-9._:\-]+)
            \s+
            (?P<input>\$?\s*[0-9]+(?:\.[0-9]+)?)
            \s+
            (?P<output>\$?\s*[0-9]+(?:\.[0-9]+)?)
            \s*$
            """,
            _re.VERBOSE,
        )

        rates = {}
        try:
            with open(rates_path, "r", encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    m = rate_line_re.match(line)
                    if not m:
                        # print(f"Warning: Unrecognized rates line at {rates_path}:{lineno}: {raw!r}", file=sys.stderr) # Removed direct print
                        self._warnings.append(f"Warning: Unrecognized rates line at {rates_path}:{lineno}: {raw!r}")
                        continue
                    mdl = m.group("model").strip().lower()

                    def _parse_money(s: str) -> float:
                        return float(s.replace("$", "").strip())

                    in_per_m = _parse_money(m.group("input"))
                    out_per_m = _parse_money(m.group("output"))
                    rates[mdl] = (in_per_m, out_per_m)
        except Exception as e:
            # print(f"Warning: Error reading site_token_rates.txt: {e}", file=sys.stderr) # Removed direct print
            self._warnings.append(f"Warning: Error reading site_token_rates.txt: {e}")
            return ""

        if not rates:
            return ""

        key = model.lower()
        if key not in rates:
            # print(f"Warning: No token rates found for model '{model}' in {rates_path}.", file=sys.stderr) # Removed direct print
            self._warnings.append(f"Warning: No token rates found for model '{model}' in {rates_path}.")
            return ""

        in_per_million, out_per_million = rates[key]
        input_cost = (prompt_tokens / 1_000_000.0) * in_per_million
        output_cost = (completion_tokens / 1_000_000.0) * out_per_million
        total_cost = input_cost + output_cost

        report = (
            f"Model: {model}\n"
            f"Input tokens: {prompt_tokens:,} @ ${in_per_million:.4f}/1M = ${input_cost:.6f}\n"
            f"Output tokens: {completion_tokens:,} @ ${out_per_million:.4f}/1M = ${output_cost:.6f}\n"
            f"---------------------------------------------\n"
            f"Estimated total cost: ${total_cost:.6f} USD"
        )
        return report

    # ---------- File commands ----------
    def delete_session_file(self, filename: str) -> str: # Modified return type
        output = []
        if filename in self.session_file_list:
            self.session_file_list.remove(filename)
            output.append(f"Unloaded local session file: {os.path.basename(filename)}")
        else:
            output.append(f"Warning: File not found in session: {os.path.basename(filename)}")
            # No cloud deletion attempt if not in local session list

        if self.provider and hasattr(self.provider, "delete_file"):
            try:
                provider_output = self.provider.delete_file(filename) # Provider handles its own cloud metadata
                if provider_output:
                    output.append(f"Provider reports: {provider_output}")
            except Exception as e:
                # print(f"Warning: could not delete cloud file {filename}: {e}", file=sys.stderr) # Removed direct print
                self._warnings.append(f"Warning: could not delete cloud file {filename}: {e}")
                output.append(f"Warning: Failed to delete cloud file {os.path.basename(filename)}: {e}")
        else:
            output.append(f"Note: No cloud provider or cloud file deletion method available for {os.path.basename(filename)}.")
        
        return "\n".join(output)

    def handle_file_commands(self, line: str) -> Optional[str]: # Modified return type
        tokens = line.split()
        output_buffer: List[str] = []

        if not tokens:
            return None

        cmd = tokens[0].lower()

        # Input / Load file command
        if cmd in ("input", "load"):
            if len(tokens) >= 2 and tokens[1].lower() == "file":
                if len(tokens) < 3:
                    output_buffer.append("Usage: input/load file <path>")
                    return "\n".join(output_buffer)
                
                file_path = " ".join(tokens[2:]) # Rejoin paths that might have spaces
                file_path = os.path.expanduser(os.path.expandvars(file_path.strip('"\'')))

                if not os.path.exists(file_path):
                    output_buffer.append(f"Error: File not found: {file_path}")
                    return "\n".join(output_buffer)

                if file_path not in self.session_file_list:
                    self.session_file_list.append(file_path)
                    output_buffer.append(f"Loaded file into session: {os.path.basename(file_path)}")
                else:
                    output_buffer.append(f"File already in session: {os.path.basename(file_path)}")
                return "\n".join(output_buffer)
            # Handle "input files <path1>, <path2>"
            elif len(tokens) >= 2 and tokens[1].lower() == "files":
                file_paths_str = " ".join(tokens[2:])
                # Robustly split by comma and handle quotes
                file_candidates = shlex.split(file_paths_str.replace(',', ' '))
                
                for candidate in file_candidates:
                    file_path = os.path.expanduser(os.path.expandvars(candidate.strip('"\'')))
                    if not os.path.exists(file_path):
                        output_buffer.append(f"Error: File not found: {file_path}")
                        continue
                    if file_path not in self.session_file_list:
                        self.session_file_list.append(file_path)
                        output_buffer.append(f"Loaded file into session: {os.path.basename(file_path)}")
                    else:
                        output_buffer.append(f"File already in session: {os.path.basename(file_path)}")
                return "\n".join(output_buffer) if output_buffer else "No files processed."


        # DELETE
        if cmd == "delete" or cmd == "remove" or cmd == "rm":
            if len(tokens) == 2 and tokens[1] == "*":
                if not self.session_file_list:
                    output_buffer.append("No session files to delete.")
                    return "\n".join(output_buffer)
                
                files_to_delete_in_session = list(self.session_file_list) # Operate on a copy
                for f_path in files_to_delete_in_session:
                    output_buffer.append(self.delete_session_file(f_path)) # Capture individual delete outputs
                output_buffer.append("Deleted all session files.")
                return "\n".join(output_buffer)

            if len(tokens) >= 2:
                target_token_start_idx = 1
                if tokens[1].lower() == "file" or tokens[1].lower() == "files":
                    target_token_start_idx = 2
                    if len(tokens) < 3:
                        output_buffer.append("Usage: delete file <name>")
                        return "\n".join(output_buffer)
                
                target_name = " ".join(tokens[target_token_start_idx:]).strip('"\'')

                # Find all matching files by basename (or full path if target is a path)
                matches = []
                for f_path in self.session_file_list:
                    if target_name == f_path or os.path.basename(f_path) == target_name:
                        matches.append(f_path)

                if not matches:
                    output_buffer.append(f"No such file in session: {target_name}")
                    return "\n".join(output_buffer)

                for f_path in matches:
                    output_buffer.append(self.delete_session_file(f_path))
                
                return "\n".join(output_buffer)

            return None # Not a recognized delete command pattern

        # LIST
        if cmd == "list":
            if len(tokens) == 1 or tokens[1].lower() in ("files", "*"):
                if not self.session_file_list:
                    output_buffer.append("No session files.")
                    return "\n".join(output_buffer)
                output_buffer.append("Current session files:")
                for f_path in sorted(self.session_file_list):
                    output_buffer.append(f"- {f_path}")
                return "\n".join(output_buffer)

            target = " ".join(tokens[1:]).strip('"\'')
            matches = [f_path for f_path in self.session_file_list if target == f_path or os.path.basename(f_path) == target]
            if not matches:
                output_buffer.append(f"No such file in session: {target}")
                return "\n".join(output_buffer)

            output_buffer.append(f"Matching session files for '{target}':")
            for f_path in sorted(matches):
                output_buffer.append(f"- {f_path}")
            return "\n".join(output_buffer)

        return None # Command not recognized as a local file command


# ---------- Provider base ----------
class ai_provider:
    def open_session(self):
        raise NotImplementedError

    def close_session(self):
        raise NotImplementedError

    def send_message(self, prompt, intro_prompt, input_file_list, delete_file_list) -> Tuple[str, int, int, int]:
        """
        Sends a message to the AI model.
        Returns a tuple: (text_response, prompt_tokens, completion_tokens, total_tokens)
        """
        raise NotImplementedError

    def delete_file(self, local_path: str) -> Optional[str]: # Modified return type
        """Optional: delete a cloud file corresponding to local_path. Returns status string."""
        # Default implementation for providers without explicit cloud file management
        return f"No cloud deletion available for '{os.path.basename(local_path)}' with this provider."


# ---------- AWS Bedrock provider ----------
class aws_bedrock_provider(ai_provider):
    def __init__(self, parent: api_eda_ai_assist):
        self.parent = parent
        self.client = None
        self.aws_access_key_id = parent.cfg.get("ASH_AWS_ACCESS_KEY_ID", None)
        self.aws_secret_access_key = parent.cfg.get("ASH_AWS_SECRET_ACCESS_KEY", None)
        self.aws_region = parent.cfg.get("ASH_AWS_REGION", "us-east-1")
        self.model = parent.cfg["ASH_MODEL"]
        # (local_path, cloud_name) - no remote storage, cloud_name is for tracking only
        self._session_files: List[tuple[str, str]] = []
        self.history: List[Dict[str, Any]] = []  # Tracks conversation turns only
        self.debug = parent.debug

    def open_session(self):
        if self.debug:
            print("open_session(aws_bedrock:%s)" % self.model)
        import boto3

        if self.aws_access_key_id and self.aws_secret_access_key:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
            )
        else:
            # Fall back to boto3 default credential chain (IAM role, ~/.aws/credentials, etc.)
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region,
            )
        self.history = []  # Reset history on new session

    def close_session(self):
        if self.debug:
            print("close_session(aws_bedrock)")
            print("Clearing in-memory session files:")
            for local_path, _ in self._session_files:
                print(f"- {local_path}")
        self._session_files.clear()
        self.history = []
        self.client = None

    def delete_file(self, local_path: str) -> str: # Modified return type
        before = list(self._session_files)
        # Filter out the file to be deleted from the provider's internal list
        self._session_files = [
            item for item in self._session_files if item[0] != local_path
        ]
        removed_count = len(before) - len(self._session_files)
        
        # It's important that parent.session_file_list is updated by api_eda_ai_assist.delete_session_file
        # so this method only needs to care about its own _session_files.

        if self.debug:
            removed = [p for p, _ in before if p not in [q for q, _ in self._session_files]]
            if removed:
                return f"aws_bedrock_provider: removed from _session_files: {removed}"
            else:
                return f"aws_bedrock_provider: file {os.path.basename(local_path)} not found in _session_files."
        
        return f"File {os.path.basename(local_path)} removed from Bedrock session tracking." if removed_count > 0 else f"File {os.path.basename(local_path)} not tracked by Bedrock session."

    def send_message(
        self, prompt, intro_prompt, input_file_list, delete_file_list
    ) -> Tuple[str, int, int, int]:
        if self.debug:
            print("send_message(aws_bedrock) %s" % prompt)

        # Detect model family
        is_anthropic = "claude" in self.model.lower()
        # is_nova = "nova" in self.model.lower() # not explicitly used for different pathing

        if delete_file_list:
            for path_to_delete in delete_file_list:
                # The delete_file method updates _session_files and parent.session_file_list
                _ = self.delete_file(path_to_delete)

        # Register any new files into the session tracking list.
        # This adds to the provider's _session_files, which in turn populates parent.session_file_list
        # via the handle_file_commands -> delete_session_file flow.
        # Ensure parent.session_file_list is up-to-date here too.
        for each_file in input_file_list:
            if each_file not in self.parent.session_file_list:
                self.parent.session_file_list.append(each_file)
            if not any(each_file == item[0] for item in self._session_files):
                name = self.parent.make_ash_cloud_name(each_file)
                self._session_files.append((each_file, name))

        # Re-read all session files fresh on every call (mirrors azure_gateway_provider behavior)
        file_blocks = []
        for each_file, name in self._session_files:
            if os.path.exists(each_file):
                try:
                    size_bytes = os.path.getsize(each_file)
                    size_mb = size_bytes / (1024 * 1024)
                    size_str = f"{size_mb:,.1f} MB"
                except Exception:
                    size_str = "unknown size"
                if self.debug:
                    print(f"reading {each_file} ({size_str})")
                try:
                    with open(each_file, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    file_blocks.append(
                        f"=== FILE: {os.path.basename(each_file)} ===\n{content}\n"
                    )
                except Exception as e:
                    # print(f"Error reading file '{each_file}': {e}", file=sys.stderr) # Removed direct print
                    self.parent._warnings.append(f"Error reading file '{each_file}': {e}")
            else:
                # print(f"Error reading file '{each_file}'") # Removed direct print
                self.parent._warnings.append(f"Error: File '{each_file}' does not exist on local disk.")


        # Assemble messages in the same order as azure_gateway_provider:
        # [file content block] -> [history] -> [current user prompt]
        # The intro_prompt is passed separately as the Bedrock "system" parameter.
        messages = []

        if file_blocks:
            files_context = "\n".join(file_blocks)
            file_message_text = (
                f"Here are my files:\n\n{files_context}\n\n"
                f"Answer questions using these files and referencing their filenames."
            )
            if is_anthropic:
                messages.append({"role": "user", "content": [{"type": "text", "text": file_message_text}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": "Understood. I have reviewed the files and am ready to answer questions."}]})
            else:
                # For non-Anthropic, prepend file context to user prompt if no direct content type support
                prompt = file_message_text + "\n\n" + prompt

        if is_anthropic:
            prompt_message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
        else: # Generic fallback for other models, assuming text-only input
            prompt_message = {"role": "user", "content": prompt}


        messages.extend(self.history) # Use extend instead of += to avoid shallow copy issues
        messages.append(prompt_message)

        try:
            import json

# Note: max_tokens is required and is at the max at 8K
            if is_anthropic:
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 8192,
                    "system": intro_prompt,
                    "messages": messages,
                }
            else:
                # Fallback for other models. Assume generic text-only chat.
                # Many Bedrock models (e.g. Cohere, AI21) don't use 'system' role
                # or have different message formats. This is a very basic fallback.
                # For more robustness, model-specific logic would be needed.
                messages_for_other_models = []
                if intro_prompt:
                    messages_for_other_models.append({"role": "system", "content": intro_prompt})
                messages_for_other_models.extend(messages) # Add existing messages (history + current prompt)

                request_body = {
                    "max_tokens": 8192,
                    "messages": messages_for_other_models,
                }

            if self.debug:
                print("invoke_model(aws_bedrock) request messages: %s" % messages)

            response = self.client.invoke_model(
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )

            response_body = json.loads(response["body"].read())

            # Extract response text 
            rts = ""
            if response_body.get("content"):
                if isinstance(response_body["content"], list) and response_body["content"] and isinstance(response_body["content"][0], dict):
                    if "text" in response_body["content"][0]:
                        rts = response_body["content"][0]["text"].strip()
                    elif "type" in response_body["content"][0]:
                        rts = "".join(
                            block.get("text", "")
                            for block in response_body["content"]
                            if block.get("type") == "text"
                        ).strip()
            elif response_body.get("completion"): # Some older Bedrock models like original Claude
                rts = response_body["completion"].strip()
            elif response_body.get("generations") and response_body["generations"][0].get("text"): # Cohere Command
                rts = response_body["generations"][0]["text"].strip()
            elif response_body.get("results") and response_body["results"][0].get("outputText"): # AI21 Jurassic
                rts = response_body["results"][0]["outputText"].strip()


            # Extract token usage 
            # This part is highly model-dependent. Using generic keys if available.
            prompt_tokens = usage.get("input_tokens", 0) if (usage := response_body.get("usage", {})) else 0
            completion_tokens = usage.get("output_tokens", 0) if usage else 0
            # Anthropic models sometimes have token counts directly in the top-level response
            if not prompt_tokens and "prompt_tokens" in response_body:
                prompt_tokens = response_body["prompt_tokens"]
            if not completion_tokens and "completion_tokens" in response_body:
                completion_tokens = response_body["completion_tokens"]
            
            total_tokens = prompt_tokens + completion_tokens

            # Append to history in correct format for model family
            if is_anthropic:
                self.history.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
                self.history.append({"role": "assistant", "content": [{"type": "text", "text": rts}]})
            else: # generic fallback
                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": rts})

            if not usage and self.debug:
                self.parent._warnings.append(
                    "Warning: Bedrock API response did not contain usage information."
                )

            return rts, prompt_tokens, completion_tokens, total_tokens

        except Exception as e:
            error_message = f"AI error (AWS Bedrock): {self.model} : {type(e).__name__}: {e}"
            # print(error_message, file=sys.stderr) # Removed direct print
            self.parent._warnings.append(error_message) # Store error as warning
            return error_message, 0, 0, 0


# ---------- Azure provider ----------
class azure_gateway_provider(ai_provider):
    def __init__(self, parent: api_eda_ai_assist):
        self.parent = parent
        self.client = None
        self.chat = None
        self.api_key = parent.cfg["ASH_API_KEY"]
        self.model = parent.cfg["ASH_MODEL"]
        # (local_path, cloud_name) - cloud_name is purely for tracking, not actual Azure object
        self._session_files: List[tuple[str, str]] = []
        self.history: List[Dict[str, str]] = [] # Azure uses its own history tracking
        self.debug = parent.debug

    def open_session(self):
        if self.debug:
            print("open_session(azure_gateway:%s)" % self.model)
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            azure_endpoint=self.parent.cfg["ASH_ENDPOINT"],
            api_key=self.parent.cfg["ASH_API_KEY"],
            api_version=self.parent.cfg["ASH_API_VERSION"],
        )
        self.chat = self.client.chat
        self.history = [] # Reset history on new session

    def close_session(self):
        if self.debug:
            print("close_session(azure_gateway)")
        self._session_files.clear() # Clear internal file list on close
        self.history.clear() # Clear chat history as well

    def delete_file(self, local_path: str) -> str: # Modified return type
        # For Azure, "cloud files" are just included in prompt content; nothing to delete remotely.
        # We only need to drop them from our internal list.
        before = list(self._session_files)
        self._session_files = [item for item in self._session_files if item[0] != local_path]
        removed_count = len(before) - len(self._session_files)

        if self.debug:
            removed = [p for p, _ in before if p not in [q for q, _ in self._session_files]]
            if removed:
                return f"azure_gateway_provider: removed from _session_files: {removed}"
            else:
                return f"azure_gateway_provider: file {os.path.basename(local_path)} not found in _session_files."
        
        return f"File {os.path.basename(local_path)} removed from Azure session tracking." if removed_count > 0 else f"File {os.path.basename(local_path)} not tracked by Azure session."


    def send_message(self, prompt, intro_prompt, input_file_list, delete_file_list) -> Tuple[str, int, int, int]:
        message_parts = [] # Collect content for current user prompt
        file_blocks = []

        if delete_file_list:
            for path_to_delete in delete_file_list:
                # The delete_file method updates _session_files and parent.session_file_list
                _ = self.delete_file(path_to_delete)

        intro_prompt_message = {"role": "system", "content": intro_prompt}
        # New 2026.02.24
        messages = [intro_prompt_message]

        # Register any new files into the session tracking list
        for each_file in input_file_list:
            if each_file not in self.parent.session_file_list:
                self.parent.session_file_list.append(each_file)
            if not any(each_file == item[0] for item in self._session_files):
                name = self.parent.make_ash_cloud_name(each_file)
                self._session_files.append((each_file, name))

        for each_file, name in self._session_files:
            if os.path.exists(each_file):
                try:
                    size_bytes = os.path.getsize(each_file)
                    size_mb = size_bytes / (1024 * 1024)
                    size_str = f"{size_mb:,.1f} MB"
                except Exception:
                    size_str = "unknown size"
                if self.debug:
                    print(f"uploading {each_file} ({size_str})")
                with open(each_file, "r", encoding="utf-8") as f:
                    content = f.read()
                file_blocks.append(f"=== FILE: {os.path.basename(each_file)} ===\n{content}\n")
            else:
                # print(f"Error reading file '{each_file}'") # Removed direct print
                self.parent._warnings.append(f"Error: File '{each_file}' does not exist on local disk.")

        files_context = "\n".join(file_blocks)
        if self._session_files:
            message_parts.append(f"Here are my files:\n\n{files_context}\n\nAnswer questions using these files and referencing their filenames.")

        message_parts.append(prompt) # Add the actual user prompt

        # Combine all parts into a single user message for the current turn
        combined_user_content = "\n".join(message_parts)
        prompt_message = {"role": "user", "content": combined_user_content}
        
        messages.extend(self.history) # Use extend
        messages.append(prompt_message)

        try:
            response = self.chat.completions.create(model=self.model, messages=messages) # Use 'messages' list
            answer = response.choices[0].message.content
            if self.debug:
                print(messages)

            # Append only the current turn's actual user prompt and assistant response to history
            self.history.append({"role": "user", "content": prompt}) # Only the user's specific text
            self.history.append({"role": "assistant", "content": answer})

            # Extract raw usage numbers
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            elif self.debug:
                # print("Warning: Azure API response did not contain usage information.", file=sys.stderr) # Removed direct print
                self.parent._warnings.append("Warning: Azure API response did not contain usage information.")


            return answer, prompt_tokens, completion_tokens, total_tokens
        except Exception as e:
            error_message = f"AI error (Azure): {type(e).__name__}: {e}"
            # print(error_message, file=sys.stderr) # Removed direct print
            self.parent._warnings.append(error_message) # Store error as warning
            return error_message, 0, 0, 0 # Return 0 tokens on error


# ---------- Gemini provider ----------
class gemini_provider(ai_provider):
    def __init__(self, parent: api_eda_ai_assist):
        self.parent = parent
        self.client = None
        self.chat = None
        self.api_key = parent.cfg["ASH_API_KEY"]
        self.model = parent.cfg["ASH_MODEL"]
        # (local_path, uploaded_file_object)
        self._session_files: List[tuple[str, object]] = []
        self.debug = parent.debug

    def open_session(self):
        if self.debug:
            print("open_session(gemini:%s)" % self.model)
        from google import genai

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client()
        self.chat = self.client.chats.create(model=self.model)

    def close_session(self):
        if self.debug:
            print("close_session(gemini)")
            print("Current Gemini cloud files:")
        for local_path, uploaded_file_obj in self._session_files:
            if self.debug:
                print(
                    f"- Name: {uploaded_file_obj.name} | Display Name: {uploaded_file_obj.display_name} | URI: {uploaded_file_obj.uri}"
                )
            try:
                self.client.files.delete(name=uploaded_file_obj.name)
                if self.debug:
                    print(f"File {local_path} deleted successfully from Gemini Files API.")
            except Exception as e:
                if self.debug:
                    # print(f"Warning: Failed to delete {local_path} (cloud name {uploaded_file_obj.name}) from Gemini Files API: {e}", file=sys.stderr) # Removed direct print
                    self.parent._warnings.append(f"Warning: Failed to delete {local_path} (cloud name {uploaded_file_obj.name}) from Gemini Files API: {e}")

        self._session_files.clear()
        self.chat = None # Invalidate chat object

    def delete_file(self, local_path: str) -> str: # Modified return type
        client = self.client
        remaining = []
        found_and_deleted = False
        output_messages = []

        for lp, obj in self._session_files:
            if lp == local_path:
                try:
                    client.files.delete(name=obj.name)
                    if self.debug:
                        output_messages.append(f"Deleted cloud file {obj.display_name} (local: {lp}) via Gemini API.")
                    else:
                        output_messages.append(f"Cloud file {os.path.basename(lp)} deleted from Gemini.")
                    found_and_deleted = True
                except Exception as e:
                    if self.debug:
                        output_messages.append(f"Warning: Failed to delete cloud file {obj.display_name} (local: {lp}): {e}")
                    else:
                        output_messages.append(f"Warning: Failed to delete cloud file {os.path.basename(lp)}: {e}")
                    remaining.append((lp, obj)) # If deletion fails, keep it in internal list
            else:
                remaining.append((lp, obj))
        self._session_files = remaining
        # The parent.session_file_list is updated by api_eda_ai_assist.delete_session_file, so no need here.

        if not found_and_deleted and not output_messages:
            return f"File {os.path.basename(local_path)} not tracked by Gemini session."
        return "\n".join(output_messages)


    def send_message(self, prompt, intro_prompt, input_file_list, delete_file_list) -> Tuple[str, int, int, int]:
        if self.debug:
            print("send_message(gemini) %s" % prompt)
        client = self.client
        chat = self.chat

        if delete_file_list:
            for path_to_delete in delete_file_list:
                # This call updates _session_files and parent.session_file_list
                _ = self.delete_file(path_to_delete)

        for each_file in input_file_list:
            if each_file not in self.parent.session_file_list:
                self.parent.session_file_list.append(each_file)
            if not any(each_file == item[0] for item in self._session_files):
                if not os.path.exists(each_file):
                    # print(f"Warning: Attempted to upload non-existent file: {each_file}", file=sys.stderr) # Removed direct print
                    self.parent._warnings.append(f"Warning: Attempted to upload non-existent file: {each_file}")
                    continue # Skip this file

                if self.debug:
                    print(f"Attempting to Upload file '{each_file}'")
                name = self.parent.make_ash_cloud_name(each_file)
                try:
                    uploaded_file = client.files.upload(
                        file=each_file,
                        config={"mime_type": "text/plain", "display_name": name},
                    )
                    if self.debug:
                        print(f"Uploaded file '{each_file}' as: {uploaded_file.name}")
                    self._session_files.append((each_file, uploaded_file))
                except Exception as e:
                    # print(f"Error uploading file '{each_file}' to Gemini: {e}", file=sys.stderr) # Removed direct print
                    self.parent._warnings.append(f"Error uploading file '{each_file}' to Gemini: {e}")


        contents = []
        contents.append(intro_prompt)
        if self._session_files: # Only add this block if there are *any* files in session
            contents.append("Here are files for analysis:")
            i = 0
            for each_file, uploaded_file in self._session_files:
                # Only include files that were explicitly part of the *current prompt's* input_file_list
                # or all session files? The original code added all _session_files regardless of input_file_list.
                # Let's stick to the original behavior and include all active session files.
                contents.append(f"File {i+1} ({os.path.basename(each_file)}):")
                contents.append(uploaded_file)
                i += 1
        contents.append(prompt)

        try:
            response = chat.send_message(contents)
            if self.debug:
                print("chat.send_message(gemini) \nQ: %s \nA: %s" % (contents, response.text.strip()))
            rts = response.text.strip()

            # Extract raw usage numbers
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if response.usage_metadata:
                total_tokens = response.usage_metadata.total_token_count
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
            elif self.debug:
                # print("Warning: Gemini API response did not contain usage information.", file=sys.stderr) # Removed direct print
                self.parent._warnings.append("Warning: Gemini API response did not contain usage information.")


            return rts, prompt_tokens, completion_tokens, total_tokens
        except Exception as e:
            error_message = f"AI error (Gemini): {type(e).__name__}: {e}"
            # print(error_message, file=sys.stderr) # Removed direct print
            self.parent._warnings.append(error_message) # Store error as warning
            return error_message, 0, 0, 0


# ---------- Platform / shell ----------
def on_windows() -> bool:
    return sys.platform.startswith("win32")


def find_powershell() -> Optional[str]:
    for candidate in ("pwsh", "powershell.exe", "powershell"):
        path = shutil.which(candidate)
        if path:
            return path
    for candidate in (
        r"C:\Program Files\PowerShell\7\pwsh.exe",
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    ):
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

    try:
        ppid = os.getppid()
        # On some systems /proc/<pid>/exe is not available or readable. Fallback to common shells.
        try:
            exe = os.readlink(f"/proc/{ppid}/exe")
            shell_path = exe
            shell_name = os.path.basename(exe)
        except (FileNotFoundError, PermissionError):
            shell_name = os.environ.get("SHELL", "bash").split('/')[-1]
            shell_path = shutil.which(shell_name) or f"/bin/{shell_name}"


        if shell_name in ("bash", "zsh", "sh"):
            return "bash", shell_path
        if shell_name in ("csh", "tcsh"):
            return "csh", shell_path

        # Default fallback if detection by ppid/env fails
        bash = shutil.which("bash")
        if bash:
            return "bash", bash
        csh_path = find_csh()
        if not csh_path:
            print("Error: `bash` or `csh` not found. Please install a shell and try again.", file=sys.stderr)
            sys.exit(127)
        return "csh", csh_path
    except Exception:
        # Generic fallback for unexpected errors
        bash = shutil.which("bash")
        if bash:
            return "bash", bash
        csh_path = find_csh()
        if not csh_path:
            print("Error: `bash` or `csh` not found. Please install a shell and try again.", file=sys.stderr)
            sys.exit(127)
        return "csh", csh_path


# ---------- Prompt ----------
def truncate_string(input_string, max_length=30):
    if len(input_string) > max_length:
        return input_string[-max_length:]
    else:
        return input_string


def format_prompt(buffering_ai, buffering_ai_ctrl_d_hint_sent) -> str:
    if buffering_ai:
        prompt = "...> "
        if not buffering_ai_ctrl_d_hint_sent:
            prompt = "(Ctrl+D to send)\n" + prompt
#       return PROMPT_COLOR + prompt + RESET_COLOR
        return prompt
    cwd = truncate_string(os.getcwd())
#   return f"{PROMPT_COLOR}[ash]:{cwd}% {RESET_COLOR}"
    return f"[ash]:{cwd}%"


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
def run_shell_command(shell_type: str, shell_path: str, command: str) -> int:
    import subprocess as _subprocess

    shell_name = os.path.basename(shell_path)

    if shell_name in ("bash", "zsh", "sh"):
        wrapped = (
            "shopt -s expand_aliases 2>/dev/null; "
            "[ -f ~/.bashrc ] && source ~/.bashrc; "
            "[ -f ~/.bash_aliases ] && source ~/.bash_aliases; "
            f"{command}"
        )
        argv = [shell_path, "-c", wrapped]

    elif shell_name in ("csh", "tcsh"):
        argv = [shell_path, "-c", command]

    elif shell_name in ("pwsh", "powershell.exe"):
        argv = [shell_path, "-NoLogo", "-Command", f"& {{ {command} }}"]

    else:
        print(f"Warning: Unknown shell type '{shell_type}', executing command directly with {shell_path}.", file=sys.stderr)
        argv = [shell_path, "-c", command]

    try:
        completed = _subprocess.run(
            argv,
            stdin=None,
            stdout=None,
            stderr=None,
# This breaks Vi being called from Ash
#           stdin=_subprocess.DEVNULL, # Ensure child process doesn't inherit stdin for non-interactive commands
#           stdout=sys.stdout,
#           stderr=sys.stderr,
            cwd=os.getcwd(),
            env=os.environ.copy(),
            text=False, # Use bytes for stdin/stdout to avoid encoding issues with raw shell output
            check=False,
        )
        return completed.returncode
    except FileNotFoundError:
        print(f"Error: Shell executable not found: {shell_path}", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        print(f"Execution error: {exc}", file=sys.stderr)
        return 1


# ---------- History & Completion ----------
def get_path_executables() -> List[str]:
    exes = set()
    path_env = os.environ.get("PATH", "")
    sep = ";" if on_windows() else ":"
    for p in path_env.split(sep):
        if not p or not os.path.isdir(p):
            continue
        try:
            for name in os.listdir(p):
                full = os.path.join(p, name)
                # Check if it's a file and executable (for POSIX) or just a file (for Windows, where .exe suffices)
                if os.path.isfile(full) and (os.access(full, os.X_OK) or on_windows()):
                    # On Windows, add common extensions if not present to match shell behavior
                    if on_windows() and not any(name.lower().endswith(ext) for ext in ('.exe', '.cmd', '.bat', '.ps1')):
                        continue
                    exes.add(name)
        except Exception:
            continue
    return sorted(exes)


def _get_history_list() -> List[str]:
    if _readline_available:
        try:
            import readline  # type: ignore

            return [
                readline.get_history_item(i + 1)
                for i in range(readline.get_current_history_length())
                if readline.get_history_item(i + 1) is not None
            ]
        except Exception:
            pass
    return INMEM_HISTORY[:]


def _init_readline():
    global _readline_available, _history_loaded, _win_completion_active
    _readline_available = False
    _history_loaded = False
    _win_completion_active = False

    try:
        import readline  # type: ignore

        _readline_available = True
    except ImportError:
        # Fallback for Windows if pyreadline3 is not installed
        if on_windows():
            try:
                import pyreadline3 as readline # type: ignore
                _readline_available = True
                _win_completion_active = True
            except ImportError:
                _readline_available = False
        else:
            _readline_available = False
    except Exception as e:
        print(f"Warning: readline initialization failed: {e}", file=sys.stderr)
        _readline_available = False

    if _readline_available:
        try:
            # Re-import to ensure it's the correct readline module (pyreadline3 or native)
            if on_windows() and not _win_completion_active: # if native readline failed but pyreadline3 might work
                import pyreadline3 as readline # type: ignore
                _win_completion_active = True
            elif not on_windows() or _win_completion_active: # if already determined to be available
                import readline # type: ignore
            else:
                raise ImportError("No suitable readline module found.")

            if os.path.exists(HISTORY_FILE):
                readline.read_history_file(HISTORY_FILE)
                _history_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load history file: {e}", file=sys.stderr)
            pass

        try:
            INMEM_HISTORY.clear()
            for i in range(readline.get_current_history_length()):
                item = readline.get_history_item(i + 1)
                if item is not None:
                    INMEM_HISTORY.append(item)
        except Exception:
            pass

        try:
            readline.set_history_length(HISTORY_LIMIT)
        except Exception:
            pass

        try:
            executables_cache = get_path_executables()
            
            # Add builtin commands for completion
            BUILTIN_COMMANDS = ["cd", "exit", "quit", "history", "status", "flush", "restart", "list", "delete", "input", "load"] # Added input, load
            
            def complete(text, state):
                buffer = readline.get_line_buffer()
                cursor = readline.get_endidx()
                
                # If completing at the very start of the line or the first token
                is_first_token = True
                try:
                    lex = shlex.split(buffer[:cursor])
                    if len(lex) > 1:
                        is_first_token = False
                    elif len(lex) == 1 and buffer[cursor-1] != ' ' and buffer[cursor-1] != '\t':
                        # If cursor is within the first token
                        is_first_token = True 
                    else: # empty or space after first token
                        is_first_token = False
                except ValueError: # Malformed input, try simpler split
                    parts = buffer[:cursor].split()
                    is_first_token = len(parts) <= 1

                candidates: List[str] = []
                
                if is_first_token:
                    candidates.extend(BUILTIN_COMMANDS)
                    candidates.extend(executables_cache)

                # File/directory completion
                pattern = os.path.expandvars(os.path.expanduser(text)) + "*"
                matches = glob.glob(pattern)
                for m in matches:
                    display = m
                    if os.path.isdir(m) and not display.endswith(os.sep):
                        display += os.sep
                    candidates.append(display)

                seen = set()
                ordered = []
                for c in sorted(candidates): # Sort for consistent order
                    if c not in seen and c.startswith(text):
                        seen.add(c)
                        ordered.append(c)

                if state < len(ordered):
                    return ordered[state]
                return None

            readline.set_completer(complete)
            if on_windows() and _win_completion_active: # pyreadline3 binds Tab automatically
                pass 
            elif not on_windows(): # native readline needs explicit binding
                readline.parse_and_bind("tab: complete")
        except Exception as e:
            print(f"Warning: Failed to set up tab completion: {e}", file=sys.stderr)
            pass

    else: # If readline is not available
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if line:
                            INMEM_HISTORY.append(line)
                _history_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load history file (no readline): {e}", file=sys.stderr)
            pass


def _save_history():
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    except Exception:
        pass

    if _readline_available:
        try:
            import readline # type: ignore
            readline.set_history_length(HISTORY_LIMIT)
            readline.write_history_file(HISTORY_FILE)
            return
        except Exception as e:
            print(f"Warning: Failed to save readline history: {e}", file=sys.stderr)
            pass

    try:
        start = max(0, len(INMEM_HISTORY) - HISTORY_LIMIT)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            for item in INMEM_HISTORY[start:]:
                f.write(item + "\n")
    except Exception as e:
        print(f"Warning: Failed to save in-memory history: {e}", file=sys.stderr)
        pass


# ---------- Bang expansion ----------
BANG_RE = re.compile(r"^!(.+)$")


def expand_bang(line: str) -> str:
    m = BANG_RE.match(line.strip())
    if not m:
        return line

    token = m.group(1).strip()
    hist = _get_history_list()
    if not hist:
        raise ValueError("event not found: history is empty")

    if token == "!":
        return hist[-1]

    if token.startswith("-"):
        try:
            n = int(token[1:])
            if n <= 0 or n > len(hist):
                raise ValueError
        except Exception:
            raise ValueError(f"bad event specification: !{token}")
        return hist[-n]

    if token.isdigit():
        idx = int(token)
        if idx <= 0 or idx > len(hist):
            raise ValueError(f"event not found: !{token}")
        return hist[idx - 1]

    prefix = token
    for cmd in reversed(hist):
        if cmd.startswith(prefix):
            return cmd

    raise ValueError(f"event not found: !{token}")


# ---------- Built-in 'history' ----------
def print_history():
    hist = _get_history_list()
    for i, cmd in enumerate(hist, start=1):
        print(f"{i:5d}  {cmd}")


# ---------- Main REPL ----------
def main():
    ai = api_eda_ai_assist()
    shell_type, shell_path = detect_shell()
    buffering_ai_ctrl_d_hint_sent = False
    buffering_ai = False
    paste_buffer: List[str] = []

    _init_readline()

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    if "--help" in sys.argv or "-h" in sys.argv:
        ai.print_help()
        return

    if "--version" in sys.argv or "-v" in sys.argv:
        ai.print_version()
        return

    if len(sys.argv) > 1:
        line = " ".join(sys.argv[1:])
        ai.open_ai_session()
        if ai.provider: # Check if session opened successfully
            rts = ai.ask_ai(line)
            print(rts)
            for w in ai.get_warnings(): 
                print(f"[warning] {w}")
            ai.print_session_status()
            ai.close_ai_session()
        else:
            print("Failed to initialize AI session for one-shot command.", file=sys.stderr)
        return

    print("-------------------------------------------------------------------------------")
    print("Hi, I'm Ash (eda_ai_assist), your cloud AI electrical-engineering assistant.   ")
    print("Within your own shell, I interpret plain-English and analyze your EDA files.   ")
    print("I became operational at Black Mesa Labs in Sammamish, WA on February 8th, 2026.")
    print("Press Ctrl+D on an empty line to enter or exit multi-line input mode.          ")
    print("-------------------------------------------------------------------------------")

    while True:
        try:
            try:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
            except Exception:
                pass
            try:
                line = input(format_prompt(buffering_ai, buffering_ai_ctrl_d_hint_sent))
                if buffering_ai:
                    buffering_ai_ctrl_d_hint_sent = True
                else:
                    buffering_ai_ctrl_d_hint_sent = False
            except EOFError:
                if not buffering_ai:
                    buffering_ai = True
                    paste_buffer = []
                    print("(Begin Buffering. Type your prompt and paste your data. Ctrl+D again to finish.)")
                    continue
                else:
                    buffering_ai = False
                    full_prompt_from_buffer = "\n".join(paste_buffer)
                    paste_buffer = []
                    if not full_prompt_from_buffer.strip():
                        print("Empty buffered input ignored.")
                        continue
                    if not ai.provider: # Open session if not already open (e.g., first AI command)
                        ai.open_ai_session()
                    if ai.provider: # Proceed only if session is active
                        rts = ai.ask_ai(full_prompt_from_buffer)
                        print(rts)
                        for w in ai.get_warnings(): 
                            print(f"[warning] {w}")
                        if not ai.provider:
                            return # Handles special hard-limit reached closed session case
                        ai.print_session_status()
                    else:
                        print("AI session is not active. Cannot process buffered input.", file=sys.stderr)
                    continue

            if not line.strip():
                continue

        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue

        if buffering_ai:
            paste_buffer.append(line)
            continue

        if line.strip():
            INMEM_HISTORY.append(line)
            if _readline_available:
                try:
                    import readline  # type: ignore
                    readline.add_history(line)
                except Exception:
                    pass

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass

        stripped = line.strip()

        if stripped in {"exit", "quit"}:
            break

        if stripped in ("restart", "restart session", "new session", "reset", "flush"):
            if ai.provider:
                ai.close_ai_session() # This also resets the global token counters and provider
            ai.open_ai_session() # This ensures a fresh session and provider
            if ai.provider:
                print("AI session restarted.")
            else:
                print("Failed to restart AI session.", file=sys.stderr)
            continue

        if stripped == "history" and not buffering_ai:
            print_history()
            continue

        if stripped == "status" and not buffering_ai:
            if not ai.provider:
                print("AI provider not initialized. Use 'restart' to open a session.")
                # ai.open_ai_session() # Don't auto-open on status, user should manage
            ai.print_session_status()
            continue

        try:
            expanded = expand_bang(line)
            if expanded != line:
                print(expanded)
            line = expanded
            stripped = line.strip()
        except ValueError as e:
            print(str(e), file=sys.stderr)
            continue

        if stripped.startswith("cd"):
            rest = stripped[len("cd"):].lstrip()
            # Handle possible shell command separators after cd argument
            for sep in (";", "&&", "||", "|", ">", "<"):
                i = rest.find(sep)
                if i != -1:
                    rest = rest[:i].strip()
                    break
            try:
                # Use shlex.split for robust parsing of quotes etc.
                toks = shlex.split(rest)
                arg = toks[0] if toks else ""
            except Exception:
                # Fallback for simpler cases if shlex fails (e.g., unbalanced quotes)
                arg = rest.split()[0] if rest else ""
            handle_cd(arg)
            continue

        # 1. Built-in file commands
        if ai.handle_file_commands(line):
            continue

        # 2. AI request
        if ai.is_ai_request(line):
            if not ai.provider: # Open session if not already open (e.g., first AI command)
                ai.open_ai_session()
            if ai.provider: # Proceed only if session is active
                rts = ai.ask_ai(line)
                print(rts)
                for w in ai.get_warnings(): 
                    print(f"[warning] {w}")
                if not ai.provider:
                    return # Handles special hard-limit reached closed session case
                ai.print_session_status()
            else:
                print("AI session is not active. Cannot process AI request.", file=sys.stderr)
            continue

        # 3. Shell command
        _ = run_shell_command(shell_type, shell_path, line)

    if ai.provider:
        ai.close_ai_session()

    _save_history()
    print("Bye.")


if __name__ == "__main__":
    main()
