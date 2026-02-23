# eda_ai_assist
Ash (AI‑Assisted Shell) wraps around your existing terminal and brings modern AI/LLM capabilities directly to the command line for analyzing EDA files.

# Ash CLI Assistant

Ash is helpful command-line AI assistant for Electrical Engineers who work with large EDA artifacts.  It wraps around your existing shell and brings modern AI/LLM capabilities directly into Linux/UNIX terminals and Windows PowerShell. Ash emphasizes clarity, predictability, and operational safety — with strict control over credentials, logging, and multi‑user behavior.

## Features
- **AI-Enabled EDA Assistance:** Natural-language processing for EDA file analysis.
- **Cross-Platform Shell Wrapper:** Works on Windows (PowerShell) and Linux/macOS (bash/zsh/csh).
- **Interactive REPL:** Persistent history, tab completion, and bash-like bang expansion (!!, !-n, !n, !prefix).
- **Multi-line Input Mode:** Press Ctrl+D on an empty line to enter or exit multi-line buffering.
- **Contextual File Handling:** Automatically detects and uploads files referenced in prompts (e.g., `file foo.vcd`).
- **Output to File:** Create output files from AI responses using "output to file <filename>" or "write to <filename>".
- **Built-in Session Commands:**
    - `flush` or `restart`: Clears AI chat history and resets token counters, starting a new session.
    - `status`: Prints current session token usage, number of Ash-managed cloud files, and last AI response time.
    - `history`: Displays command history.
    - `exit` or `quit`: Exits the wrapper.
- **Built-in File Management Commands (for Ash-managed files):**
    - `list` / `list files` / `list *`: Show all files currently tracked by Ash in the session.
    - `list <filename>`: Show specific matching session files.
    - `delete <filename>` / `delete file <filename>`: Delete matching session files and their cloud counterparts.
    - `delete *`: Delete all Ash-managed session files.
- **Token Usage Monitoring:** Tracks AI token usage per session with warnings at 2,500,000 and 4,000,000 tokens, and automatic session termination at 5,000,000 tokens to prevent runaway costs.
- **Cost Estimation:** Estimates session costs based on a configurable `site_token_rates.txt`.
- **Flexible Configuration:** Supports multiple credential delivery paths, including secure encrypted tokens and raw API keys, with site-wide defaults and environment variable overrides.
- **Provider Abstraction:** Supports Gemini (default) and Azure Gateway models.
- **Structured Logging:** Detailed usage logging and per-user accounting.
- **Operational Safety:** Predictable, auditable behavior; works offline except for model calls.

## Installation
Clone the repository and install dependencies:

```bash
python3 -m pip install google-genai openai
```
Ensure the `google-genai` package is installed for Gemini support and `openai` for Azure Gateway support.

## API Key Configuration and Environment Variables
Ash's behavior is configured via environment variables and site files within `ASH_DIR`. Variables are applied in the following precedence: 1) Internal defaults, 2) `site_defaults.txt`, 3) Explicit `ASH_*` user environment variables.

### Important Environment Variables:
- `ASH_DIR`: Base directory for site configuration (default: `~/.ash`).
- `ASH_PROVIDER`: AI provider name (default: `gemini`).
- `ASH_MODEL`: Model name for the provider (default: `gemini-2.0-flash`).
- `ASH_ENDPOINT`: Provider API endpoint.
- `ASH_API_VERSION`: Provider API version.
- `ASH_API_KEY`: Raw API key (optional if `ASH_USER_TOKEN` is used).
- `ASH_USER_TOKEN`: Encrypted API token (site‑managed).
- `ASH_USER_PROMPT`: Optional prefix added to all AI requests.
- `ASH_LOG_DIR`: Directory for usage logs (default: `ASH_DIR`).
- `ASH_LOG_IDENTITY`: Determines log identity (`username` or `process`).

### Credential Delivery Paths:
Ash supports three ways to provide credentials, evaluated in this order:

1.  **ASH_USER_TOKEN (Encrypted Token):**
    A secure, multi-user-safe encrypted token. This token is bound to a username and protected with an HMAC signature. It requires a `site_key.txt` in `ASH_DIR` to decrypt.
    Environment variable:
    ```bash
    export ASH_USER_TOKEN="username|cipher|signature"
    ```

2.  **ASH_API_KEY (Raw API Key):**
    A simple raw API key for single-user environments. Overrides the encrypted token if both are present.
    ```bash
    export ASH_API_KEY="your-api-key"
    ```

3.  **GEMINI_API_KEY (SDK Fallback):**
    If neither Ash-specific variable is set, the Gemini SDK will fall back to its standard environment variable. (Note: Azure Gateway requires `ASH_API_KEY`).
    ```bash
    export GEMINI_API_KEY="your-api-key"
    ```

### Site Files (located in `ASH_DIR`):
- `site_prompt.txt`: Optional site-wide prompt preface added to each AI query.
- `site_defaults.txt`: Key=value lines to override defaults (quoted values OK).
- `site_key.txt`: Secret used to decrypt `ASH_USER_TOKEN` (keep secure).
- `site_token_rates.txt`: Optional pricing file (`<model> <input_per_1M> <output_per_1M>`).
- `site_billing.txt`: Optional account billing information displayed with help.
- `site_restrictions.txt`: Optional policy text shown to new users with help.

---

## Cross‑Platform Setup (Windows + Linux)

Ash stores its configuration in a per‑user directory called **ASH_DIR**.
For single‑user installs, the recommended locations are:

- **Linux:** `$HOME/.ash`
- **Windows:** `%USERPROFILE%\.ash`

Below are example setups for both environments.

---

## Windows Setup (ash.bat)

Copy `eda_ai_assist.py` into:

```
%USERPROFILE%\.ash
```

Example `ash.bat`:

```bat
@echo off
REM Configure Ash for a single-user Windows install
setx ASH_DIR "%USERPROFILE%\.ash"
setx ASH_API_KEY "mykey123"
setx ASH_PROVIDER "gemini"
setx ASH_MODEL "gemini-2.0-flash"

REM Execute the assistant from the configured directory
python "%USERPROFILE%\.ash\eda_ai_assist.py"
```

This script works when launched from **CMD or PowerShell**, because:
- `setx` writes variables to the Windows user environment (visible to both shells)
- `%USERPROFILE%` expands correctly inside `.bat` files
- No shell‑specific syntax is used inside the Python command

---

## Linux Setup

Add the following to your shell startup file (`~/.bashrc`, `~/.zshrc`, or `~/.cshrc`) when installing into:

```
$HOME/.ash
```

### Bash / Zsh

```bash
export ASH_DIR="$HOME/.ash"
export ASH_API_KEY="mykey123"
export ASH_PROVIDER="gemini"
export ASH_MODEL="gemini-2.0-flash"
# Add a convenient 'ash' command
alias ash="$ASH_DIR/eda_ai_assist.py"
```

### CSH / TCSH

```csh
setenv ASH_DIR "$HOME/.ash"
setenv ASH_API_KEY "mykey123"
setenv ASH_PROVIDER "gemini"
setenv ASH_MODEL "gemini-2.0-flash"
# Add a convenient 'ash' command
alias ash "$ASH_DIR/eda_ai_assist.py"
```

### Make sure the script is executable:
```bash
chmod +x $HOME/.ash/eda_ai_assist.py
```

## Usage
Ash can be used in one-shot mode or as an interactive shell wrapper.

### Interactive REPL:
Simply type `ash` to enter the interactive mode.
```
[ash]:/home/user%
```
- Commands that match executables run in the system shell.
- Natural-language requests are routed to the AI engine. To force AI behavior, prefix your prompt with "ash".
- Press `Ctrl+D` on an empty line to enter/exit multi-line mode for longer prompts.

**Examples:**
```
[ash]:/home/khubbard% summarize file top.v
[ash]:/home/khubbard% cd ../simulations
[ash]:/home/khubbard/simulations% analyze simulation1.vcd and find when in time the signal fault_int asserts. Output to file results.txt
Created results.txt (1.2 MB)
[ash]:/home/khubbard/simulations% list
results.txt
simulation1.vcd
top.v
[ash]:/home/khubbard/simulations% status
[status] Tokens: 125K | Files: 3 (23.5MB) | Time: 4.56s
[ash]:/home/khubbard/simulations% flush
AI session restarted.
```

### One-Shot Mode:
Provide your prompt directly as arguments to `ash`.

```bash
ash "Explain clock domain crossing"
```
```bash
ash "Summarize the worst setup timing violations in file post_route_timing.rpt"
```
```bash
ash "analyze simulation1.vcd and find when in time the signal fault_int asserts. Output to file results.txt"
```

### Using Ash as a Python API:
You can integrate Ash's AI capabilities into your Python programs.

```python
from eda_ai_assist import api_eda_ai_assist

ash = api_eda_ai_assist()
ash.open_ai_session() # Initialize the AI session
response, warnings = ash.ask_ai("Summarize this file top.sdc")
print(response)
for w in warnings:
    print(f"[warning] {w}")
ash.close_ai_session() # Clean up the AI session
```

## Security Model
Ash is designed for environments where multiple users may share a system. Key security features include:

- Encrypted tokens bound to usernames.
- HMAC integrity verification to prevent tampering.
- Site secret key (`site_key.txt`) required for decryption.
- Raw API key override for single-user setups.
- Clear error handling when credentials are missing or invalid.

## Logging
Ash maintains two logs in the directory specified by `ASH_LOG_DIR` (defaults to `ASH_DIR`):

### usage_queries.log
An append-only log of all prompts sent to the AI, along with timestamps, user/process identity, model, and token usage. Safe under concurrent writes.

### usage_totals.log
Tracks cumulative per-user (or per-process) upload/download token totals. This file is written atomically to avoid corruption.

## License
This project is released under the GPLv3 license. See `LICENSE` for details.
