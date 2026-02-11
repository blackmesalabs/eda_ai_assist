```text
File `eda_ai_assist.py` implements `api_eda_ai_assist`, a command-line tool (Ash) for natural-language analysis of EDA (Electronic Design Automation) files.  It allows users to interact with AI models (currently focusing on Gemini) using natural language prompts to analyze and manipulate EDA data.

**Key Features:**

*   **AI Interaction:**  The core functionality revolves around the `ask_ai` function, which handles user prompts, interacts with the AI model, and presents the results.  It utilizes the Google Gemini API for AI processing. The `is_ai_request` function determines if a user prompt warrants AI processing, looking for trigger words like "how," "why," "analyze," etc.
*   **Configuration:**  Loads configurations from environment variables (e.g., `ASH_API_KEY`, `ASH_MODEL`, `ASH_PROVIDER`, `ASH_DIR`), site-specific defaults (`site_defaults.txt`), and hardcoded defaults, with environment variables taking highest priority. The configuration includes secrets, models, log paths, and prompts.
*   **File Handling:** It can process multiple input files, specified in a natural language way, using the `ai_input_files` function, which identifies existing files mentioned in the prompt. It allows users to direct AI output to a file using commands like `"output to file foo.txt"` via the `ai_output_file` function.  It verifies the existence and textual nature of these files.
*   **Prompt Engineering:**  Constructs a prompt for the AI model by combining a site-specific prompt (`site_prompt.txt`), user's input, and the content of input files. It has built in functions to handle user prompts, extract model names from the prompt, and warn user on very large prompts.
*   **Usage Logging:** Tracks user activity, logging usage statistics (timestamps, usernames, model used, API key, upload/download bytes) to `usage_queries.log` and aggregates totals in `usage_totals.log`. Handles user agreements through `site_restrictions.txt`, requiring confirmation before processing if the file exists and the user hasn't agreed yet.
*   **Security:** Obfuscates API keys when logging. It can decrypt tokens using a site-specific secret key for secure API key management.

*   **Command Line Interface (CLI):** Acts as a shell wrapper with command history, tab completion, and supports bash-like bang (!) history expansion, similar to bash. Detects the user's shell (PowerShell on Windows, csh on Linux/macOS), executes commands, and intercepts the `cd` command.
*   **History and Tab Completion:** Maintains a persistent command history saved to `~/.shell_wrapper_history`. Tab completion is supported via readline (POSIX) or pyreadline3 (Windows, if available).  Bang expansion allows users to re-execute previous commands.
*   **Help and Version Information:** The script includes built-in functions to print version and help information, including site-specific billing and restrictions.
*   **Platform Agnostic:** Designed to work on both Windows and POSIX-compliant operating systems.

**How it Works:**

1.  The script initializes, detects the shell environment, and loads configuration data.
2.  It presents a prompt to the user.
3.  The user enters a command.
4.  The script checks if the command is a built-in (e.g., `exit`, `history`, `cd`) or an AI request.
5.  If it's an AI request, it constructs a prompt, interacts with the AI model, logs usage, presents the results, and handles file input/output as specified.
6.  Otherwise, it executes the command using the detected shell.
7.  The script saves the command history and repeats the process until the user exits.

**Important Considerations:**

*   Requires a valid API key to access the AI model. The key should be stored securely, ideally using the encrypted token mechanism.
*   Depends on external libraries, notably `google-generativeai` for Gemini integration and optionally `readline` or `pyreadline3` for shell features.
*   Uses site-specific configuration files (`site_prompt.txt`, `site_defaults.txt`, `site_restrictions.txt`, `site_key.txt`) to customize behavior and enforce usage policies.

Overall, `eda_ai_assist.py` provides a powerful interface for using AI to analyze EDA data, integrating natural language processing with shell-like interaction and robust configuration management.
```
