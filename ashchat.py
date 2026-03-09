#!/usr/bin/env python3
########################################################################
# ashchat.py - Combined AshChat module
# Merged from: ashchat_main.py, ashchat_core.py, ashchat_wx.py, ashchat_tk.py
#
# Copyright (C) 2026  Kevin M. Hubbard BlackMesaLabs
# Licensed under GNU General Public License v3 or later
########################################################################
import os
import sys
import threading
import queue
import re
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Dict, Any, Optional, List, Callable

try:
    import wx
except ImportError:
    wx = None

try:
    from eda_ai_assist import api_eda_ai_assist
except ImportError:
    print("Error: Could not import eda_ai_assist module.")
    sys.exit(1)



# ============= ashchat_core.py =============

import os
import sys
import threading
import queue



class AshChatCore:
    def __init__(self, gui_callbacks: Dict[str, Callable]):
        self.gui_callbacks = gui_callbacks
        self.ai = api_eda_ai_assist()
        self.ai.get_env_config()

        self.loaded_files: List[str] = []
        self.model_config: Dict[str, Dict[str, Any]] = self._load_site_model_list()
        self.current_model_nickname = None
        self.current_model_selected = False

        self.output_directory = self._get_default_output_directory()
        self.current_output_dir_var = self.output_directory

        self.message_queue: queue.Queue = queue.Queue()
        self.ai_thread: Optional[threading.Thread] = None
        self.is_running = True

        self._initialize_model()

    def _get_default_output_directory(self) -> str:
        """Return the OS user Downloads folder as default output directory."""
        if sys.platform.startswith("win32"):
            return os.path.join(os.path.expanduser("~"), "Downloads")
        else:
            return os.path.join(os.path.expanduser("~"), "Downloads")

    def _load_site_model_list(self) -> Dict[str, Dict[str, Any]]:
        """Load model configuration from ASH_DIR/site_model_list.txt or use defaults."""
        config = {}
        ash_dir = os.environ.get("ASH_DIR", os.path.expanduser("~/.ash"))

        site_model_list_path = os.path.join(ash_dir, "site_model_list.txt")

        try:
            with open(site_model_list_path, "r", encoding="utf-8") as f:
                current_section = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        current_section = line[1:-1].strip()
                        config[current_section] = {}
                    elif "=" in line and current_section:
                        key, val = line.split("=", 1)
                        val = val.strip().strip('"').strip("'")
                        config[current_section][key.strip()] = val.strip()
        except Exception as e:
            print(f"Error loading model config: {e}", file=sys.stderr)

        return config

    def _initialize_model(self):
        """
        Initialize model by matching environment config to site_model_list.txt.
        
        Priority order:
        1. Match ASH_PROVIDER and ASH_MODEL from environment to a configured model
        2. If no match, use first model from site_model_list.txt
        3. Open AI session with selected model
        """
        env_provider = self.ai.cfg.get("ASH_PROVIDER", "").lower()
        env_model = self.ai.cfg.get("ASH_MODEL", "").lower()

        matched_nickname = None

        # Step 1: Try to find matching provider+model combo in site_model_list.txt
        if env_provider and env_model and self.model_config:
            for nickname, config in self.model_config.items():
                config_provider = config.get("ASH_PROVIDER", "").lower()
                config_model = config.get("ASH_MODEL", "").lower()

                if config_provider == env_provider and config_model == env_model:
                    matched_nickname = nickname
                    break

        # Step 2: If match found, use it. Otherwise use first model.
        if matched_nickname:
            self.current_model_nickname = matched_nickname
            selected_config = self.model_config[matched_nickname]
        elif self.model_config:
            self.current_model_nickname = list(self.model_config.keys())[0]
            selected_config = self.model_config[self.current_model_nickname]
        else:
            # No site_model_list.txt; use environment defaults
            selected_config = {}

        # Step 3: Update AI config and open session
        if selected_config:
            self.ai.cfg.update(selected_config)

        self.ai.open_ai_session()
        self.current_model_selected = True

    def switch_model(self, nickname: str):
        """Switch to a different configured model."""
        if nickname in self.model_config:
            config = self.model_config[nickname]
            if self.ai.provider:
                self.ai.close_ai_session()
            self.ai.cfg.update(config)
            self.ai.open_ai_session()
            self.current_model_nickname = nickname
            self.gui_callbacks["update_title"](
                config.get("ASH_PROVIDER", "unknown"),
                config.get("ASH_MODEL", "unknown")
            )

    def load_file(self, file_path: str):
        """Load a file for analysis"""
        if not file_path:
            return

        if not os.path.exists(file_path):
            self.gui_callbacks["show_error_message"](f"File not found: {file_path}")
            return

        abs_path = os.path.abspath(file_path)

        if abs_path in self.loaded_files:
            self.gui_callbacks["show_warning_message"](
                f"File already loaded: {os.path.basename(abs_path)}",
                "Already Loaded"
            )
            return

        try:
            size_bytes = os.path.getsize(abs_path)
            size_mb = size_bytes / (1024 * 1024)
            if size_mb > 100:
                if not self.gui_callbacks["ask_confirmation"](
                    f"File is {size_mb:.1f} MB. Loading large files may be slow. Continue?",
                    "Large File Warning"
                ):
                    return

            self.loaded_files.append(abs_path)
            self.gui_callbacks["update_loaded_files_display"]()
            self.gui_callbacks["update_status"]()
            self.gui_callbacks["show_info_message"](
                f"Loaded: {os.path.basename(abs_path)}"
            )
        except Exception as e:
            self.gui_callbacks["show_error_message"](f"Error loading file: {e}")

    def unload_files(self, file_paths: List[str]):
        """Unload one or more files from the session."""
        for file_path in file_paths:
            if file_path in self.loaded_files:
                self.loaded_files.remove(file_path)
            if file_path in self.ai.session_file_list:
                self.ai.session_file_list.remove(file_path)

        self.gui_callbacks["update_loaded_files_display"]()

    def _rewrite_prompt_with_output_dir(self, prompt: str) -> str:
        """
        Rewrite prompt to replace output/write directives with full paths.
        Matches "output to filename" or "write to filename" and substitutes
        the full path (output_directory joined with filename).
        """
        import re

        pattern = r"\b(?:output|write)\s+to\s+(\S+)"
        matches = list(re.finditer(pattern, prompt, re.IGNORECASE))

        if not matches:
            return prompt

        match = matches[0]
        after_match = prompt[match.end():].strip()
        tokens = after_match.split()

        if not tokens:
            return prompt

        filename = tokens[0].strip('"\',;:!?()[]{}')
        full_path = os.path.join(self.output_directory, filename)

        before_match = prompt[:match.start()]
        after_replacement = prompt[match.end() + len(filename):]

        rewritten = f"{before_match}output to {full_path}{after_replacement}"

        return rewritten

    def send_message(self, prompt: str):
        """Queue a message to be sent to the AI after output directory rewrite."""
        if not prompt.strip():
            return

        rewritten_prompt = self._rewrite_prompt_with_output_dir(prompt)

        enhanced_prompt = ""
        if self.loaded_files:
            for file_path in self.loaded_files:
                enhanced_prompt += f"file \"{file_path}\"\n"
        enhanced_prompt += rewritten_prompt

        self.message_queue.put(("send_message", enhanced_prompt))

    def _send_command_to_ai(self, command: str, append_to_chat: bool = True, disable_controls: bool = True):
        """Queue an internal command for processing."""
        self.message_queue.put(("send_command", command, append_to_chat, disable_controls))

    def process_queue(self):
        """Process all pending messages from the queue."""
        while not self.message_queue.empty():
            try:
                task = self.message_queue.get_nowait()

                if task[0] == "send_message":
                    prompt = task[1]
                    self._process_send_message(prompt)
                elif task[0] == "send_command":
                    command = task[1]
                    append_to_chat = task[2] if len(task) > 2 else True
                    disable_controls = task[3] if len(task) > 3 else True
                    self._process_send_command(command, append_to_chat, disable_controls)
            except queue.Empty:
                break
            except Exception as e:
                self.gui_callbacks["show_error_message"](f"Error processing queue: {e}")


    def _process_send_message(self, prompt: str):
        """Process a user message: send to AI and display response."""
        self.gui_callbacks["disable_controls"]()
        self.gui_callbacks["append_chat"]("user", prompt)
 
        if not self.ai.provider:
            self.gui_callbacks["append_chat"]("system", "Error: AI provider not initialized.")
            self.gui_callbacks["re_enable_controls"]()
            self.gui_callbacks["update_status"]()
            return



        # Run AI call in background thread
        thread = threading.Thread(target=self._ai_response_worker, args=(prompt,))
        thread.daemon = True
        thread.start()

    def _ai_response_worker(self, prompt: str):
        """Background thread worker for AI response."""
        try:
            response = self.ai.ask_ai(prompt)
            self.gui_callbacks["append_chat"]("assistant", response)

            for warning in self.ai.get_warnings():
                self.gui_callbacks["append_chat"]("system", f"[warning] {warning}")

            if not self.ai.provider:
                self.gui_callbacks["append_chat"]("system", "AI session was closed due to token limits.")

        except Exception as e:
            self.gui_callbacks["append_chat"]("system", f"Error: {e}")

        finally:
            self.gui_callbacks["clear_input_text"]()
            self.gui_callbacks["re_enable_controls"]()
            self.gui_callbacks["update_status"]()

 
    def _process_send_command(self, command: str, append_to_chat: bool, disable_controls: bool):
        """Process an internal command (file operations, etc.)."""
        if disable_controls:
            self.gui_callbacks["disable_controls"]()

        if append_to_chat:
            self.gui_callbacks["append_chat"]("user", command)

        try:
            response = self.ai.handle_file_commands(command)
            if response:
                self.gui_callbacks["append_chat"]("system", response)
                self.gui_callbacks["update_loaded_files_display"]()
            else:
                if not self.ai.provider:
                    self.gui_callbacks["append_chat"]("system", "Error: AI provider not initialized.")
                    if disable_controls:
                        self.gui_callbacks["re_enable_controls"]()
                    self.gui_callbacks["update_status"]()
                    return

                try:
                    response = self.ai.ask_ai(command)
                    self.gui_callbacks["append_chat"]("assistant", response)

                    for warning in self.ai.get_warnings():
                        self.gui_callbacks["append_chat"]("system", f"[warning] {warning}")

                except Exception as e:
                    self.gui_callbacks["append_chat"]("system", f"Error: {e}")

        except Exception as e:
            self.gui_callbacks["append_chat"]("system", f"Error: {e}")

        if append_to_chat:
            self.gui_callbacks["clear_input_text"]()

        if disable_controls:
            self.gui_callbacks["re_enable_controls"]()

        self.gui_callbacks["update_status"]()

    def flush_session(self):
        """Flush session: close AI, clear files, reset output directory, restart."""
        if self.ai.provider:
            self.ai.close_ai_session()

        self.loaded_files.clear()
        self.ai.session_file_list.clear()
        self.output_directory = self._get_default_output_directory()
        self.current_output_dir_var = self.output_directory

        self.ai.open_ai_session()
        self.gui_callbacks["clear_chat_display"]()
        self.gui_callbacks["update_loaded_files_display"]()
        self.gui_callbacks["append_chat"]("system", "Session flushed. AI session restarted.")
        self.gui_callbacks["update_status"]()

    def close_ai_session(self):
        """Close the AI session."""
        if self.ai.provider:
            self.ai.close_ai_session()

    def _format_token_count(self, count: int) -> str:
        """Format token count with K/M/G suffixes."""
        if count == 0:
            return "0"
        if count < 1000:
            return str(count)
        if count < 1_000_000:
            return f"{count // 1000}K"
        if count < 1_000_000_000:
            return f"{count // 1_000_000}M"
        return f"{count // 1_000_000_000}G"

    def _truncate_path(self, path: str, max_length: int = 30) -> str:
        """Truncate path from left with .. prefix if longer than max_length."""
        if len(path) <= max_length:
            return path
        return ".." + path[-(max_length - 2):]

    def get_status_text(self) -> str:
        """Return status bar text with token count, response time, input files, and output directory."""
        total_tokens = self.ai.token_cnt_total
        response_time_sec = round(self.ai.last_response_time)
        num_files = len(self.loaded_files)
        output_dir_truncated = self._truncate_path(self.output_directory, 20)

        token_str = self._format_token_count(total_tokens)

        return f"Tokens: {token_str} | Response Time: {response_time_sec}s | Input Files: {num_files} | Output Dir: {output_dir_truncated}"

    def get_help_text(self) -> str:
        """Return help text for the application."""
        return """
AshChat Help

File Management:
  Load File (Ctrl+L): Load a file for analysis
  Unload File (Ctrl+U): Remove selected files from session
  Output Directory: Automatically set to the directory of the last loaded file

Output Files:
  Use "output to filename.txt" or "write to filename.txt" in your prompts
  Files will be saved to the current output directory

Chat Operations:
  Flush Session: Clear history, unload all files, reset token counters
  Send Message (Ctrl+Enter in multi-line mode, or Enter in single-line mode)

Keyboard Shortcuts:
  Ctrl+L: Load File
  Ctrl+U: Unload File
  Ctrl+S: Save Selected Text
  Ctrl+Shift+S: Save Chat Transcript
  Ctrl+D: Toggle multi-line input mode
  Enter (single-line): Send message
  Ctrl+Enter (multi-line): Send message
  F1: Show this help

Font Size:
  Use Format menu to adjust text size (8pt to 18pt)
  Default is 10pt

Multi-line Mode:
  Press Ctrl+D on an empty line to enter or exit multi-line buffering
  Useful for pasting large blocks of text or typing multiple lines
"""

    def get_about_text(self) -> str:
        """Return about text for the application."""
        return """
AshChat v1.01
EDA AI Assistant GUI Frontend

Part of eda_ai_assist project
Repository: https://github.com/blackmesalabs/eda_ai_assist

Copyright (C) 2026 Kevin M. Hubbard, Black Mesa Labs

Licensed under GNU General Public License v3 or later
See https://www.gnu.org/licenses/ for details

AshChat is a graphical interface to Ash (eda_ai_assist),
a cloud AI assistant for electrical engineering.

Ash became operational at Black Mesa Labs in Sammamish, WA
on February 8th, 2026.

Built with:
  - Python 3.8+
  - Tkinter (cross-platform) or wxPython (Windows)
  - Google Generative AI (Gemini), Azure OpenAI, or AWS Bedrock

For support and updates, visit:
  https://github.com/blackmesalabs/eda_ai_assist
"""

    def get_exit_message(self) -> str:
        """Return the exit confirmation message."""
        return "Are you sure you want to exit AshChat?"


# ============= ashchat_wx.py =============


if wx is not None:
    import wx
    import sys
    import os
    
    
    
    class WxFrontend(wx.Frame):
        def __init__(self, parent, title):
            super(WxFrontend, self).__init__(parent, title=title, size=(900, 700))
            self.SetMinSize((600, 400))
    
            self.default_font_size = 10 if sys.platform.startswith("win32") else 11
            self.default_gui_font_size = 10 if sys.platform.startswith("win32") else 11
            self.current_font_size = self.default_font_size
            self.current_gui_font_size = self.default_gui_font_size
    
            self.mono_font = wx.Font(self.current_font_size, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
            self.mono_font_bold = wx.Font(self.current_font_size, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
            self.gui_font = wx.Font(self.current_gui_font_size, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
            self.gui_font_bold = wx.Font(self.current_gui_font_size, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
            self.font_updatable_widgets = []
            self.input_multiline_expanded = False
    
            gui_callbacks = {
                "show_error_message": lambda msg: wx.MessageBox(msg, "Error", wx.OK | wx.ICON_ERROR),
                "show_warning_message": lambda msg, title: wx.MessageBox(msg, title, wx.OK | wx.ICON_WARNING),
                "show_info_message": lambda msg: wx.MessageBox(msg, "Info", wx.OK | wx.ICON_INFORMATION),
                "ask_confirmation": lambda msg, title: wx.MessageBox(msg, title, wx.YES_NO | wx.ICON_QUESTION) == wx.YES,
                "update_title": self._update_title,
                "append_chat": self._append_chat,
                "update_status": self._update_status,
                "update_loaded_files_display": self._update_loaded_files_display,
                "clear_input_text": lambda: self.input_text.Clear(),
                "disable_controls": self._disable_controls,
                "re_enable_controls": self._re_enable_controls,
                "clear_chat_display": lambda: self.chat_display.Clear(),
                "revert_model_selection": self._revert_model_selection_wx,
            }
            self.core = AshChatCore(gui_callbacks)
    
            self._setup_menu()
            self._setup_panels()
            self._update_title(self.core.ai.cfg.get("ASH_PROVIDER", "unknown"), self.core.ai.cfg.get("ASH_MODEL", "unknown"))
            self.status_bar.SetStatusText(self.core.get_status_text())
    
            self.queue_timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self._process_queue_wx, self.queue_timer)
            self.queue_timer.Start(100)
    
            self.Centre()
            self.Show(True)
            self.Bind(wx.EVT_CLOSE, self._on_exit)
            self._update_font_menu_selection()
    
            # Check the current model after everything is initialized
            wx.CallAfter(self._check_current_model)
            wx.CallAfter(self._initial_file_list_refresh)
    
        def _check_current_model(self):
            """Ensure the current model is checked in the menu."""
            if self.core.current_model_nickname and self.core.current_model_nickname in self.model_radio_group:
                for item in self.all_model_items:
                    item.Check(False)
                self.model_radio_group[self.core.current_model_nickname].Check(True)
    
        def _initial_file_list_refresh(self):
            self.core._send_command_to_ai("list files", append_to_chat=False, disable_controls=False)
    
        def _update_title(self, provider: str, model: str):
            self.SetTitle(f"AshChat - {provider}:{model}")
    
        def _revert_model_selection_wx(self):
            for item in self.all_model_items:
                item.Check(False)
            if self.core.current_model_nickname and self.core.current_model_nickname in self.model_radio_group:
                self.model_radio_group[self.core.current_model_nickname].Check(True)
    
        def _on_model_select(self, nickname):
            """Handle model selection from AI menu."""
            for item in self.all_model_items:
                item.Check(False)
            self.model_radio_group[nickname].Check(True)
            self.selected_model = nickname
            self.core.switch_model(nickname)
    
        def _setup_menu(self):
            menubar = wx.MenuBar()
    
            file_menu = wx.Menu()
            file_menu.Append(wx.ID_OPEN, "&Load File\tCtrl+L", "Load a file for analysis")
            self.Bind(wx.EVT_MENU, self._load_file_wx, id=wx.ID_OPEN)
            file_menu.Append(wx.ID_DELETE, "&Unload File\tCtrl+U", "Unload selected file(s)")
            self.Bind(wx.EVT_MENU, self._unload_file_wx, id=wx.ID_DELETE)
            file_menu.AppendSeparator()
            file_menu.Append(wx.ID_SAVEAS, "&Save Chat Transcript\tCtrl+Shift+S", "Save the entire chat history")
            self.Bind(wx.EVT_MENU, self._save_transcript_wx, id=wx.ID_SAVEAS)
            file_menu.Append(wx.ID_SAVE, "Save &Selected Text\tCtrl+S", "Save selected text from chat history")
            self.Bind(wx.EVT_MENU, self._save_selected_wx, id=wx.ID_SAVE)
            file_menu.AppendSeparator()
            file_menu.Append(wx.ID_EXIT, "E&xit", "Terminate AshChat")
            self.Bind(wx.EVT_MENU, self._on_exit, id=wx.ID_EXIT)
            menubar.Append(file_menu, "&File")
    
            edit_menu = wx.Menu()
            edit_menu.Append(wx.ID_COPY, "&Copy\tCtrl+C", "Copy selected text")
            self.Bind(wx.EVT_MENU, self._copy_text_wx, id=wx.ID_COPY)
            edit_menu.Append(wx.ID_PASTE, "&Paste\tCtrl+V", "Paste text into input area")
            self.Bind(wx.EVT_MENU, self._paste_text_wx, id=wx.ID_PASTE)
            edit_menu.Append(wx.ID_SELECTALL, "Select &All\tCtrl+A", "Select all text in chat history")
            self.Bind(wx.EVT_MENU, self._select_all_wx, id=wx.ID_SELECTALL)
            edit_menu.AppendSeparator()
            edit_menu.Append(wx.ID_CLEAR, "&Clear Chat", "Clear all chat history")
            self.Bind(wx.EVT_MENU, self._clear_chat_wx, id=wx.ID_CLEAR)
            menubar.Append(edit_menu, "&Edit")
    
            format_menu = wx.Menu()
            font_sizes = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            self.font_size_menu_items = {}
            for size in font_sizes:
                menu_item = format_menu.AppendRadioItem(wx.ID_ANY, f"Font Size {size}pt", f"Set font size to {size}pt")
                self.Bind(wx.EVT_MENU, lambda evt, s=size: self._set_font_size(s), menu_item)
                self.font_size_menu_items[size] = menu_item
            format_menu.AppendSeparator()
            reset_item = format_menu.Append(wx.ID_RESET, "Reset to Default", "Reset font size to default")
            self.Bind(wx.EVT_MENU, self._reset_font_size, id=wx.ID_RESET)
            menubar.Append(format_menu, "F&ormat")
    
            ai_menu = wx.Menu()
            self.model_radio_group = {}
            self.all_model_items = []
            self.selected_model = self.core.current_model_nickname
    
            if self.core.model_config:
                providers = {}
                for nickname, config in self.core.model_config.items():
                    provider = config.get("ASH_PROVIDER", "unknown")
                    if provider not in providers:
                        providers[provider] = []
                    providers[provider].append(nickname)
    
                for provider_name in sorted(providers.keys()):
                    provider_submenu = wx.Menu()
                    for nickname in sorted(providers[provider_name]):
                        menu_item = provider_submenu.Append(wx.ID_ANY, nickname, f"Switch to {nickname} model", wx.ITEM_CHECK)
                        self.model_radio_group[nickname] = menu_item
                        self.all_model_items.append(menu_item)
                        
                        handler_id = menu_item.GetId()
                        self.Bind(wx.EVT_MENU, lambda evt, n=nickname: self._on_model_select(n), id=handler_id)
    
                    ai_menu.AppendSubMenu(provider_submenu, provider_name)
            else:
                ai_menu.Append(wx.ID_ANY, "(No models configured)", "No AI models found", wx.ITEM_NORMAL).Enable(False)
    
            ai_menu.AppendSeparator()
            flush_session_menu_item = ai_menu.Append(wx.ID_ANY, "&Flush Session", "Clear history, unload files, reset tokens")
            self.Bind(wx.EVT_MENU, self._flush_session_wx, flush_session_menu_item)
    
            send_message_menu_item = ai_menu.Append(wx.ID_ANY, "&Send Message\tCtrl+Enter", "Send the current prompt")
            self.Bind(wx.EVT_MENU, self._send_message_wx, send_message_menu_item)
    
            menubar.Append(ai_menu, "&AI")
    
            help_menu = wx.Menu()
            help_menu.Append(wx.ID_HELP, "View &Help\tF1", "Show quick reference guide")
            self.Bind(wx.EVT_MENU, self._show_help_wx, id=wx.ID_HELP)
            help_menu.AppendSeparator()
            help_menu.Append(wx.ID_ABOUT, "&About AshChat", "Show information about AshChat")
            self.Bind(wx.EVT_MENU, self._show_about_wx, id=wx.ID_ABOUT)
            menubar.Append(help_menu, "&Help")
    
            self.SetMenuBar(menubar)
    
        def _update_font_menu_selection(self):
            for size, item in self.font_size_menu_items.items():
                item.Check(False)
            if self.current_font_size in self.font_size_menu_items:
                self.font_size_menu_items[self.current_font_size].Check(True)
    
        def _setup_panels(self):
            main_sizer = wx.BoxSizer(wx.VERTICAL)
    
            splitter = wx.SplitterWindow(self)
            splitter.SetMinimumPaneSize(40)
            main_sizer.Add(splitter, 1, wx.EXPAND)
    
            # First splitter: Files and Chat
            splitter1 = wx.SplitterWindow(splitter)
            splitter1.SetMinimumPaneSize(40)
    
            # Files panel
            files_panel = wx.Panel(splitter1)
            files_box = wx.StaticBox(files_panel, label="Loaded Files (managed by Ash AI session)")
            files_box.SetFont(self.mono_font)
            self.font_updatable_widgets.append(files_box)
    
            files_sizer = wx.StaticBoxSizer(files_box, wx.VERTICAL)
    
            self.loaded_files_listbox = wx.ListBox(
                files_panel,
                style=wx.LB_SINGLE
            )
            self.loaded_files_listbox.SetFont(self.mono_font)
            self.font_updatable_widgets.append(self.loaded_files_listbox)
            files_sizer.Add(self.loaded_files_listbox, 1, wx.EXPAND | wx.ALL, 5)
            files_panel.SetSizer(files_sizer)
    
            # Chat panel
            chat_panel = wx.Panel(splitter1)
            chat_sizer = wx.BoxSizer(wx.VERTICAL)
            chat_label = wx.StaticText(chat_panel, label="Chat History")
            chat_label.SetFont(self.mono_font)
            self.font_updatable_widgets.append(chat_label)
            chat_sizer.Add(chat_label, 0, wx.ALIGN_LEFT | wx.LEFT | wx.TOP, 5)
    
            self.chat_display = wx.TextCtrl(
                chat_panel,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.VSCROLL | wx.TE_WORDWRAP
            )
            self.chat_display.SetFont(self.mono_font)
            self.chat_display.SetBackgroundColour(wx.Colour("#f0f0f0"))
            self.font_updatable_widgets.append(self.chat_display)
            chat_sizer.Add(self.chat_display, 1, wx.EXPAND | wx.ALL, 5)
            chat_panel.SetSizer(chat_sizer)
    
            self.chat_display.Bind(wx.EVT_CONTEXT_MENU, self._on_chat_display_context_menu)
    
            # Split files and chat horizontally with gravity 1/6 for files, 1/6 for chat
            splitter1.SplitHorizontally(files_panel, chat_panel)
            splitter1.SetSashGravity(0.5)
    
            # Input panel
            input_panel = wx.Panel(splitter)
            input_sizer = wx.BoxSizer(wx.VERTICAL)
            self.input_label = wx.StaticText(input_panel, label="Prompt (Enter to send, Ctrl+D for multi-line mode)")
            self.input_label.SetFont(self.mono_font)
            self.font_updatable_widgets.append(self.input_label)
            input_sizer.Add(self.input_label, 0, wx.ALIGN_LEFT | wx.LEFT | wx.TOP, 5)
    
            self.input_text = wx.TextCtrl(
                input_panel,
                style=wx.TE_MULTILINE | wx.TE_WORDWRAP | wx.VSCROLL
            )
            self.input_text.SetMinSize((-1, 40))
            self.input_text.SetFont(self.mono_font)
            self.font_updatable_widgets.append(self.input_text)
            input_sizer.Add(self.input_text, 1, wx.EXPAND | wx.ALL, 5)
            self.input_text.Bind(wx.EVT_KEY_DOWN, self._on_input_char_wx)
    
            input_panel.SetSizer(input_sizer)
    
            # Split files/chat and input vertically with gravity 2/3 for input, 1/3 for files/chat
            splitter.SplitHorizontally(splitter1, input_panel)
            splitter.SetSashGravity(0.3333)
    
            self.status_bar = self.CreateStatusBar()
            sb_font = self.status_bar.GetFont()
            sb_font.SetPointSize(self.current_gui_font_size)
            self.status_bar.SetFont(sb_font)
            self.font_updatable_widgets.append(self.status_bar)
    
            self.SetSizer(main_sizer)
            self.Layout()
    
        def _update_status(self):
            self.status_bar.SetStatusText(self.core.get_status_text())
    
        def _append_chat(self, role: str, message: str):
            if role == "user":
                self.chat_display.AppendText(f"\n[You]: {message}\n")
            elif role == "assistant":
                self.chat_display.AppendText(f"\n[Ash]: {message}\n")
            elif role == "system":
                self.chat_display.AppendText(f"\n[System]: {message}\n")
            self.chat_display.ShowPosition(self.chat_display.GetLastPosition())
    
        def _update_loaded_files_display(self):
            self.loaded_files_listbox.Clear()
            for file_path in sorted(self.core.loaded_files):
                file_name = os.path.basename(file_path)
                self.loaded_files_listbox.Append(file_name)
    #           self.loaded_files_listbox.Append(file_path)
    
        def _update_input_label(self):
            """Update the input label based on multi-line mode."""
            if self.input_multiline_expanded:
                label_text = "Multi-Line Prompt (Ctrl+Enter to send, Ctrl+D for single-line mode)"
            else:
                label_text = "Prompt (Enter to send, Ctrl+D for multi-line mode)"
            self.input_label.SetLabel(label_text)
    
        def _on_input_char_wx(self, event):
            keycode = event.GetKeyCode()
            if keycode == wx.WXK_RETURN and event.CmdDown():
                self._send_message_wx()
                return
            if keycode == wx.WXK_RETURN:
                if event.ShiftDown():
                    event.Skip()
                    return
                else:
                    if not self.input_multiline_expanded:
                        self._send_message_wx()
                        return
                    else:
                        event.Skip()
                        return
            if keycode == ord('D') and event.CmdDown():
                self._toggle_multiline_wx()
                return
            event.Skip()
    
        def _send_message_wx(self, event=None):
            prompt = self.input_text.GetValue().strip()
            self.core.send_message(prompt)
    
        def _disable_controls(self):
            if hasattr(self, 'input_text'):
                self.input_text.Disable()
    
        def _re_enable_controls(self):
            if hasattr(self, 'input_text'):
                wx.CallAfter(self.input_text.Enable)
                wx.CallAfter(self.input_text.SetFocus)
    
        def _process_queue_wx(self, event=None):
            self.core.process_queue()
    
        def _toggle_multiline_wx(self):
            if not self.input_multiline_expanded:
                self.input_text.SetMinSize((-1, 200))
                self.input_multiline_expanded = True
            else:
                self.input_text.SetMinSize((-1, 40))
                self.input_multiline_expanded = False
            self._update_input_label()
            self.Layout()
    
        def _set_font_size(self, size: int):
            if 8 <= size <= 18:
                self.current_font_size = size
                self.current_gui_font_size = size
    
                self.mono_font.SetPointSize(size)
                self.mono_font_bold.SetPointSize(size)
                self.gui_font.SetPointSize(size)
                self.gui_font_bold.SetPointSize(size)
    
                for widget in self.font_updatable_widgets:
                    if isinstance(widget, wx.StaticBox):
    #                   widget.SetFont(self.gui_font_bold)
                        widget.SetFont(self.mono_font)
                    elif isinstance(widget, wx.StaticText) or isinstance(widget, wx.Button):
                        original_font = widget.GetFont()
                        if original_font.GetWeight() == wx.FONTWEIGHT_BOLD:
    #                       widget.SetFont(self.gui_font_bold)
                            widget.SetFont(self.mono_font)
                        else:
    #                       widget.SetFont(self.gui_font)
                            widget.SetFont(self.mono_font)
                    elif isinstance(widget, wx.TextCtrl) or isinstance(widget, wx.ListBox):
                        widget.SetFont(self.mono_font)
                    elif isinstance(widget, wx.StatusBar):
                        sb_font = widget.GetFont()
                        sb_font.SetPointSize(size)
                        widget.SetFont(sb_font)
    
                self.GetMenuBar().SetFont(self.gui_font)
                self.Layout()
                self._update_font_menu_selection()
    
        def _reset_font_size(self, event=None):
            self.current_font_size = self.default_font_size
            self.current_gui_font_size = self.default_gui_font_size
            self._set_font_size(self.default_font_size)
    
        def _load_file_wx(self, event=None):
            with wx.FileDialog(
                self, "Select a file to load",
                wildcard="All files (*.*)|*.*|Text files (*.txt)|*.txt|Verilog (*.v)|*.v|VCD (*.vcd)|*.vcd",
                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
            ) as fileDialog:
                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return
                file_path = fileDialog.GetPath()
                self.core.load_file(file_path)
    
        def _unload_file_wx(self, event=None):
            if not self.core.loaded_files:
                wx.MessageBox("No files to unload.", "Info", wx.OK | wx.ICON_INFORMATION)
                return
    
            current_display_paths = list(self.core.loaded_files)
    
            with wx.MultiChoiceDialog(
                self, "Select file(s) to unload:", "Unload File",
                current_display_paths
            ) as dialog:
                if dialog.ShowModal() == wx.ID_OK:
                    selections = dialog.GetSelections()
                    if selections:
                        selected_full_paths_for_core = [current_display_paths[i] for i in selections]
                        self.core.unload_files(selected_full_paths_for_core)
    
        def _save_transcript_wx(self, event=None):
            with wx.FileDialog(
                self, "Save Chat Transcript",
                wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                defaultFile="chat_transcript.txt"
            ) as fileDialog:
                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return
                file_path = fileDialog.GetPath()
                try:
                    content = self.chat_display.GetValue()
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    wx.MessageBox(f"Chat transcript saved to {file_path}", "Success", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Failed to save transcript: {e}", "Error", wx.OK | wx.ICON_ERROR)
    
        def _save_selected_wx(self, event=None):
            selected_text = self.chat_display.GetStringSelection()
            if not selected_text:
                wx.MessageBox("Please select text to save.", "No Selection", wx.OK | wx.ICON_WARNING)
                return
    
            with wx.FileDialog(
                self, "Save Selected Text",
                wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                defaultFile="selected_text.txt"
            ) as fileDialog:
                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return
                file_path = fileDialog.GetPath()
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(selected_text)
                    wx.MessageBox(f"Selected text saved to {file_path}", "Success", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Failed to save selected text: {e}", "Error", wx.OK | wx.ICON_ERROR)
    
        def _copy_text_wx(self, event=None):
            selected_text = self.chat_display.GetStringSelection()
            if selected_text:
                if wx.TheClipboard.Open():
                    try:
                        wx.TheClipboard.SetData(wx.TextDataObject(selected_text))
                        wx.TheClipboard.Close()
                    except Exception as e:
                        if wx.TheClipboard.IsOpened():
                            wx.TheClipboard.Close()
                        wx.MessageBox(f"Failed to copy to clipboard: {e}", "Copy Error", wx.OK | wx.ICON_ERROR)
                else:
                    wx.MessageBox("Failed to open clipboard.", "Clipboard Error", wx.OK | wx.ICON_ERROR)
            else:
                wx.MessageBox("Please select text in the Chat History to copy.", "No Selection", wx.OK | wx.ICON_WARNING)
    
        def _on_chat_display_context_menu(self, event):
            menu = wx.Menu()
            copy_item = menu.Append(wx.ID_COPY, "Copy Selected Text")
            copy_item.Enable(bool(self.chat_display.GetStringSelection()))
            self.Bind(wx.EVT_MENU, self._copy_text_wx, copy_item)
            paste_item = menu.Append(wx.ID_PASTE, "Paste")
            self.Bind(wx.EVT_MENU, self._paste_text_wx, paste_item)
            self.PopupMenu(menu)
            menu.Destroy()
    
        def _paste_text_wx(self, event=None):
            if wx.TheClipboard.Open():
                if wx.TheClipboard.IsSupported(wx.DataFormat(wx.DF_TEXT)):
                    tdo = wx.TextDataObject()
                    wx.TheClipboard.GetData(tdo)
                    text = tdo.GetText()
                    self.input_text.WriteText(text)
                else:
                    wx.MessageBox("Clipboard does not contain text.", "Error", wx.OK | wx.ICON_WARNING)
                wx.TheClipboard.Close()
            else:
                wx.MessageBox("Failed to open clipboard.", "Error", wx.OK | wx.ICON_ERROR)
    
        def _select_all_wx(self, event=None):
            self.chat_display.SetSelection(-1, -1)
    
        def _clear_chat_wx(self, event=None):
            if wx.MessageBox("Clear all chat history?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
                self.chat_display.Clear()
    
        def _flush_session_wx(self, event=None):
            self.core.flush_session()
    
        def _show_scrollable_dialog_wx(self, title: str, text: str, width: int = 80, height: int = 30):
            dialog = wx.Dialog(self, title=title, size=(width * 8, height * 14))
            dialog_sizer = wx.BoxSizer(wx.VERTICAL)
    
            text_ctrl = wx.TextCtrl(
                dialog, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.VSCROLL | wx.TE_WORDWRAP,
                size=(width * 8 - 20, height * 14 - 100)
            )
            text_ctrl.SetFont(self.mono_font)
            text_ctrl.SetValue(text)
            dialog_sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALL, 5)
    
            close_button = wx.Button(dialog, wx.ID_CANCEL, "Close")
            close_button.SetFont(self.gui_font)
            dialog_sizer.Add(close_button, 0, wx.ALIGN_CENTER | wx.BOTTOM, 5)
            close_button.Bind(wx.EVT_BUTTON, lambda evt: dialog.EndModal(wx.ID_CANCEL))
    
            dialog.SetSizer(dialog_sizer)
            dialog.Centre()
            dialog.ShowModal()
            dialog.Destroy()
    
        def _show_help_wx(self, event=None):
            self._show_scrollable_dialog_wx("Help - AshChat", self.core.get_help_text(), width=90, height=35)
    
        def _show_about_wx(self, event=None):
            self._show_scrollable_dialog_wx("About AshChat", self.core.get_about_text(), width=70, height=18)
    
        def _on_exit(self, event=None):
            msg = self.core.get_exit_message()
            if wx.MessageBox(msg, "Exit", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
                self.core.close_ai_session()
                self.Destroy()
    
    
    class AshChatApp(wx.App):
        def OnInit(self):
            frame = WxFrontend(None, title="AshChat")
            return True
    
    
    def main_gui():
        app = AshChatApp()
        app.MainLoop()
    
    
    if __name__ == "__main__":
        main_gui()
    

# ============= ashchat_tk.py =============

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import sys
import os



class TkFrontend(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AshChat")
        self.geometry("900x700")
        self.minsize(600, 400)

        self.default_font_size = 10
        self.default_gui_font_size = 10
        self.current_font_size = self.default_font_size
        self.current_gui_font_size = self.default_gui_font_size

        self.input_multiline_expanded = False
        self.font_updatable_widgets = []

        gui_callbacks = {
            "show_error_message": lambda msg: messagebox.showerror("Error", msg),
            "show_warning_message": lambda msg, title: messagebox.showwarning(title, msg),
            "show_info_message": lambda msg: messagebox.showinfo("Info", msg),
            "ask_confirmation": lambda msg, title: messagebox.askyesno(title, msg),
            "update_title": self._update_title,
            "append_chat": self._append_chat,
            "update_status": self._update_status,
            "update_loaded_files_display": self._update_loaded_files_display,
            "clear_input_text": self._clear_input_text,
            "disable_controls": self._disable_controls,
            "re_enable_controls": self._re_enable_controls,
            "clear_chat_display": self._clear_chat_display,
            "revert_model_selection": self._revert_model_selection_tk,
        }
        self.core = AshChatCore(gui_callbacks)

        self._setup_menu()
        self._setup_panels()
        self._update_title(self.core.ai.cfg.get("ASH_PROVIDER", "unknown"), self.core.ai.cfg.get("ASH_MODEL", "unknown"))
        self.status_bar.config(text=self.core.get_status_text())

        self.queue_timer_id = None
        self._process_queue_tk()

        self.protocol("WM_DELETE_WINDOW", self._on_exit)

#       self.tk.call("wm", "iconphoto", self._w)

        self.after(100, self._check_current_model)
        self.after(200, self._initial_file_list_refresh)


    def _initial_file_list_refresh(self):
        self.core._send_command_to_ai("list files", append_to_chat=False, disable_controls=False)

    def _update_title(self, provider: str, model: str):
        self.title(f"AshChat - {provider}:{model}")

    def _check_current_model(self):
        if self.core.current_model_nickname and self.core.current_model_nickname in self.model_radio_group:
            for item in self.all_model_items:
                item.set(False)
            self.model_radio_group[self.core.current_model_nickname].set(True)

    def _revert_model_selection_tk(self):
        for item in self.all_model_items:
            item.set(False)
        if self.core.current_model_nickname and self.core.current_model_nickname in self.model_radio_group:
            self.model_radio_group[self.core.current_model_nickname].set(True)

    def _on_model_select(self, nickname):
        for item in self.all_model_items:
            item.set(False)
        self.model_radio_group[nickname].set(True) 
        self.core.switch_model(nickname)

    def _on_chat_display_context_menu(self, event):
        """Show context menu on right-click in chat display."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Copy", command=self._copy_text_tk)
        menu.add_command(label="Paste", command=self._paste_text_tk)
        try:
            menu.post(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _setup_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load File", command=self._load_file_tk, accelerator="Ctrl+L")
        self.bind("<Control-l>", lambda e: self._load_file_tk())
        file_menu.add_command(label="Unload File", command=self._unload_file_tk, accelerator="Ctrl+U")
        self.bind("<Control-u>", lambda e: self._unload_file_tk())
        file_menu.add_separator()
        file_menu.add_command(label="Save Chat Transcript", command=self._save_transcript_tk, accelerator="Ctrl+Shift+S")
        self.bind("<Control-Shift-S>", lambda e: self._save_transcript_tk())
        file_menu.add_command(label="Save Selected Text", command=self._save_selected_tk, accelerator="Ctrl+S")
        self.bind("<Control-s>", lambda e: self._save_selected_tk())
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Copy", command=self._copy_text_tk, accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=self._paste_text_tk, accelerator="Ctrl+V")
        edit_menu.add_command(label="Select All", command=self._select_all_tk, accelerator="Ctrl+A")
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Chat", command=self._clear_chat_tk)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        format_menu = tk.Menu(menubar, tearoff=0)
        self.font_size_var = tk.IntVar(value=self.current_font_size)
        self.font_size_menu_items = {}
        for size in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
            self.font_size_menu_items[size] = format_menu.add_radiobutton(
                label=f"Font Size {size}pt",
                variable=self.font_size_var,
                value=size,
                command=lambda s=size: self._set_font_size(s)
            )
        format_menu.add_separator()
        format_menu.add_command(label="Reset to Default", command=self._reset_font_size)
        menubar.add_cascade(label="Format", menu=format_menu)

        ai_menu = tk.Menu(menubar, tearoff=0)
        self.model_radio_group = {}
        self.all_model_items = []
        self.selected_model = self.core.current_model_nickname

        if self.core.model_config:
            providers = {}
            for nickname, config in self.core.model_config.items():
                provider = config.get("ASH_PROVIDER", "unknown")
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(nickname)

            for provider_name in sorted(providers.keys()):
                provider_submenu = tk.Menu(ai_menu, tearoff=0)
                for nickname in sorted(providers[provider_name]):
                    model_var = tk.BooleanVar()
                    menu_item = provider_submenu.add_checkbutton(
                        label=nickname,
                        variable=model_var,
                        command=lambda n=nickname: self._on_model_select(n)
                    )
                    self.model_radio_group[nickname] = model_var
                    self.all_model_items.append(model_var)

                ai_menu.add_cascade(label=provider_name, menu=provider_submenu)
        else:
            ai_menu.add_command(label="(No models configured)", state=tk.DISABLED)

        ai_menu.add_separator()
        ai_menu.add_command(label="Flush Session", command=self._flush_session_tk)
        ai_menu.add_command(label="Send Message", command=self._send_message_tk, accelerator="Ctrl+Enter")
        self.bind("<Control-Return>", lambda e: self._send_message_tk())
        menubar.add_cascade(label="AI", menu=ai_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="View Help", command=self._show_help_tk)
        self.bind("<F1>", lambda e: self._show_help_tk())
        help_menu.add_separator()
        help_menu.add_command(label="About AshChat", command=self._show_about_tk)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _setup_panels(self):
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Create main paned window for all three sections (files, chat, input)
        main_paned = tk.PanedWindow(main_frame, orient=tk.VERTICAL, sashwidth=5)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Files panel
        files_frame = tk.LabelFrame(main_paned, text="Loaded Files (managed by Ash AI session)")
        files_scrollbar = tk.Scrollbar(files_frame)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.loaded_files_listbox = tk.Listbox(files_frame, yscrollcommand=files_scrollbar.set)
        self.loaded_files_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        files_scrollbar.config(command=self.loaded_files_listbox.yview)
        self.font_updatable_widgets.append(self.loaded_files_listbox)
        main_paned.add(files_frame)

        # Chat panel
        chat_frame = tk.LabelFrame(main_paned, text="Chat History")
        chat_scrollbar = tk.Scrollbar(chat_frame)
        chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display = tk.Text(chat_frame, state=tk.NORMAL, wrap=tk.WORD, yscrollcommand=chat_scrollbar.set)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.bind("<Button-3>", self._on_chat_display_context_menu)
        chat_scrollbar.config(command=self.chat_display.yview)
        self.font_updatable_widgets.append(self.chat_display)
        main_paned.add(chat_frame)

        # Input panel
        input_frame = tk.LabelFrame(main_paned, text="Prompt (Enter to send, Ctrl+D for multi-line mode)")
        input_scrollbar = tk.Scrollbar(input_frame)
        input_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD, yscrollcommand=input_scrollbar.set)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_text.bind("<Return>", self._on_input_char_tk)
        self.input_text.bind("<Control-d>", lambda e: self._toggle_multiline_tk())
        self.input_text.bind("<Button-3>", self._on_chat_display_context_menu)
        input_scrollbar.config(command=self.input_text.yview)
        self.font_updatable_widgets.append(self.input_text)
        main_paned.add(input_frame)

        self.input_frame = input_frame

        # Status bar
        self.status_bar = tk.Label(self, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, padx=0, pady=0)
        self.font_updatable_widgets.append(self.status_bar)

    def not_setup_panels(self):
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Create main paned window for all three sections (files, chat, input)
        main_paned = tk.PanedWindow(main_frame, orient=tk.VERTICAL, sashwidth=5)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Files panel
        files_frame = tk.LabelFrame(main_paned, text="Loaded Files (managed by Ash AI session)")
        self.loaded_files_listbox = tk.Listbox(files_frame, height=6)
        self.loaded_files_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.font_updatable_widgets.append(self.loaded_files_listbox)
        main_paned.add(files_frame)

        # Chat panel
        chat_frame = tk.LabelFrame(main_paned, text="Chat History")
        self.chat_display = tk.Text(chat_frame, state=tk.NORMAL, wrap=tk.WORD)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.bind("<Button-3>", self._on_chat_display_context_menu)
        self.font_updatable_widgets.append(self.chat_display)
        main_paned.add(chat_frame)


        # Input panel
        input_frame = tk.LabelFrame(main_paned, text="Prompt (Enter to send, Ctrl+D for multi-line mode)")
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_text.bind("<Return>", self._on_input_char_tk)
        self.input_text.bind("<Control-d>", lambda e: self._toggle_multiline_tk())
        self.input_text.bind("<Button-3>", self._on_chat_display_context_menu)
        self.font_updatable_widgets.append(self.input_text)
        main_paned.add(input_frame)

        self.input_frame = input_frame

        # Status bar
        self.status_bar = tk.Label(self, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, padx=0, pady=0)
        self.font_updatable_widgets.append(self.status_bar)

    def _update_status(self):
        self.status_bar.config(text=self.core.get_status_text())

    def _append_chat(self, role: str, message: str):
        self.chat_display.config(state=tk.NORMAL)
        if role == "user":
            self.chat_display.insert(tk.END, f"\n[You]: {message}\n")
        elif role == "assistant":
            self.chat_display.insert(tk.END, f"\n[Ash]: {message}\n")
        elif role == "system":
            self.chat_display.insert(tk.END, f"\n[System]: {message}\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _update_loaded_files_display(self):
        self.loaded_files_listbox.delete(0, tk.END)
        for file_path in sorted(self.core.loaded_files):
            file_name = os.path.basename(file_path)
            self.loaded_files_listbox.insert(tk.END, file_name)
#           self.loaded_files_listbox.insert(tk.END, file_path)

    def _on_input_char_tk(self, event):
        if event.state & 0x4:
            if event.keysym == "Return":
                self._send_message_tk()
                return "break"
        if event.state & 0x1:
            if event.keysym == "Return":
                return
        else:
            if not self.input_multiline_expanded:
                self._send_message_tk()
                return "break"
        return

    def _send_message_tk(self, event=None):
        prompt = self.input_text.get("1.0", tk.END).strip()
        self.core.send_message(prompt)
        self._clear_input_text()

    def _disable_controls(self):
        if hasattr(self, 'input_text'):
            self.input_text.config(state=tk.DISABLED)

    def _re_enable_controls(self):
        if hasattr(self, 'input_text'):
            self.input_text.config(state=tk.NORMAL)
            self.input_text.focus()

    def _clear_input_text(self):
        self.input_text.delete("1.0", tk.END)

    def _process_queue_tk(self):
        self.core.process_queue()
        self.queue_timer_id = self.after(100, self._process_queue_tk)

    def _toggle_multiline_tk(self):
        if not self.input_multiline_expanded:
            self.input_frame.config(text="Multi-Line Prompt (Ctrl+Enter to send, Ctrl+D for single-line mode)")
            self.input_multiline_expanded = True
        else:
            self.input_frame.config(text="Prompt (Enter to send, Ctrl+D for multi-line mode)")
            self.input_multiline_expanded = False
        return "break"

    def _set_font_size(self, size: int):
        if 8 <= size <= 18:
            self.current_font_size = size
            self.current_gui_font_size = size

            mono_font = ("Courier", size)
            gui_font = ("Helvetica", size)

            for widget in self.font_updatable_widgets:
                if isinstance(widget, (tk.Text, tk.Listbox)):
                    widget.config(font=mono_font)
                elif isinstance(widget, tk.Label):
                    widget.config(font=gui_font)

            self.font_size_var.set(size)

    def _reset_font_size(self, event=None):
        self.current_font_size = self.default_font_size
        self.current_gui_font_size = self.default_gui_font_size
        self._set_font_size(self.default_font_size)

    def _load_file_tk(self, event=None):
        file_path = filedialog.askopenfilename(
            title="Select a file to load",
            filetypes=[("All files", "*.*"), ("Text files", "*.txt"), ("Verilog", "*.v"), ("VCD", "*.vcd")]
        )
        if file_path:
            self.core.load_file(file_path)

    def _unload_file_tk(self, event=None):
        if not self.core.loaded_files:
            messagebox.showinfo("Info", "No files to unload.")
            return

        file_list = sorted(self.core.loaded_files)

        unload_window = tk.Toplevel(self)
        unload_window.title("Unload File")
        unload_window.geometry("400x300")

        tk.Label(unload_window, text="Select file(s) to unload:").pack(padx=5, pady=5)

        listbox = tk.Listbox(unload_window, selectmode=tk.MULTIPLE)
        listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for file_path in file_list:
            listbox.insert(tk.END, file_path)

        def unload_selected():
            selections = listbox.curselection()
            if selections:
                selected_paths = [file_list[i] for i in selections]
                self.core.unload_files(selected_paths)
            unload_window.destroy()

        button_frame = tk.Frame(unload_window)
        button_frame.pack(padx=5, pady=5)
        tk.Button(button_frame, text="Unload", command=unload_selected).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=unload_window.destroy).pack(side=tk.LEFT, padx=5)

    def _save_transcript_tk(self, event=None):
        file_path = filedialog.asksaveasfilename(
            title="Save Chat Transcript",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            defaultextension=".txt"
        )
        if file_path:
            try:
                content = self.chat_display.get("1.0", tk.END)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Chat transcript saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save transcript: {e}")

    def _save_selected_tk(self, event=None):
        try:
            selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            messagebox.showwarning("No Selection", "Please select text to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Selected Text",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            defaultextension=".txt"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(selected_text)
                messagebox.showinfo("Success", f"Selected text saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save selected text: {e}")

    def _copy_text_tk(self, event=None):
        try:
            selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.clipboard_clear()
            self.clipboard_append(selected_text)
            self.update()
        except tk.TclError:
            messagebox.showwarning("No Selection", "Please select text in Chat History to copy.")

    def _paste_text_tk(self, event=None):
        try:
            text = self.clipboard_get()
            self.input_text.insert(tk.INSERT, text)
        except tk.TclError:
            messagebox.showerror("Clipboard Error", "Failed to access clipboard.")

    def _select_all_tk(self, event=None):
        self.chat_display.tag_add(tk.SEL, "1.0", tk.END)
        self.chat_display.mark_set(tk.INSERT, "1.0")
        self.chat_display.see(tk.INSERT)

    def _clear_chat_tk(self, event=None):
        if messagebox.askyesno("Confirm", "Clear all chat history?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)

    def _clear_chat_display(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _flush_session_tk(self, event=None):
        self.core.flush_session()

    def _show_scrollable_dialog_tk(self, title: str, text: str):
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry("640x480")

        text_widget = tk.Text(dialog, wrap=tk.WORD, state=tk.NORMAL)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_widget.insert("1.0", text)
        text_widget.config(state=tk.DISABLED)

        close_button = tk.Button(dialog, text="Close", command=dialog.destroy)
        close_button.pack(padx=5, pady=5)

        dialog.transient(self)
        dialog.grab_set()

    def _show_help_tk(self, event=None):
        self._show_scrollable_dialog_tk("Help - AshChat", self.core.get_help_text())

    def _show_about_tk(self, event=None):
        self._show_scrollable_dialog_tk("About AshChat", self.core.get_about_text())

    def _on_exit(self, event=None):
        msg = self.core.get_exit_message()
        if messagebox.askyesno("Exit", msg):
            self.core.close_ai_session()
            if self.queue_timer_id:
                self.after_cancel(self.queue_timer_id)
            self.destroy()


def main_gui():
    app = TkFrontend()
    app.mainloop()


if __name__ == "__main__":
    main_gui()


# ============= ashchat_main.py =============

import sys
import os

def detect_gui_toolkit():
    """Detect which GUI toolkit to use based on platform."""
    if sys.platform.startswith("win32"):
        try:
            import wx
            return "wx"
        except ImportError:
            print("Error: wxPython not found on Windows.")
            print("Please install wxPython: pip install wxPython")
            sys.exit(1)
    else:
        try:
            import tkinter as tk
            return "tk"
        except ImportError:
            print("Error: Tkinter not found.")
            print("Tkinter is usually included with Python.")
            sys.exit(1)


def main():
    """Main entry point for AshChat."""
    gui_toolkit = detect_gui_toolkit()

    if gui_toolkit == "wx":
        try:
            from ashchat_wx import main_gui as main_gui_wx
            main_gui_wx()
        except ImportError as e:
            print(f"Error: Could not import ashchat_wx module: {e}")
            sys.exit(1)
    else:
        try:
            from ashchat_tk import main_gui as main_gui_tk
            main_gui_tk()
        except ImportError as e:
            print(f"Error: Could not import ashchat_tk module: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
