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
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# This is an offshoot of SUMP3 : https://github.com/blackmesalabs/sump3
# Repository : https://github.com/blackmesalabs/eda_ai_assist
# The technical name is eda_ai_assist but the ChatBot is known as Ash.
########################################################################
import sys
import os
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple

# Conditional imports for GUI toolkits
if sys.platform.startswith("win32"):
    try:
        import wx
        # A flag to indicate which GUI is active
        USE_WXPYTHON = True
    except ImportError:
        print("Error: wxPython not found. Please install wxPython for Windows.")
        sys.exit(1)
else:
    try:
        import tkinter as tk
        from tkinter import scrolledtext, filedialog, messagebox
        USE_WXPYTHON = False
    except ImportError:
        print("Error: Tkinter not found. Tkinter is usually included with Python.")
        sys.exit(1)

# Import the eda_ai_assist API
try:
    from eda_ai_assist import api_eda_ai_assist
except ImportError:
    print("Error: Could not import eda_ai_assist module. Ensure eda_ai_assist.py is in the same directory.")
    sys.exit(1)


# --- Core Logic Class ---
class AshChatCore:
    def __init__(self, gui_callbacks: Dict[str, Any]):
        self.gui_callbacks = gui_callbacks
        self.model_config = self._load_model_config()
        self.current_model_nickname = None

        self.ai = api_eda_ai_assist()
        self.ai.open_ai_session()
        if not self.ai.provider:
            self.gui_callbacks["show_error_message"]("Failed to initialize AI session.")
            sys.exit(1)

        self.current_model_nickname = self._get_current_model_nickname()
        # Reinstated local loaded_files list to manage GUI display and prompt construction
        self.loaded_files: List[str] = [] 
        self.response_queue = queue.Queue()

        # Initial population of loaded files will be triggered by GUI after setup,
        # so removed the direct call here.

    def _load_model_config(self) -> Dict:
        """Load model configuration from site_model_list.txt"""
        ash_dir = os.environ.get("ASH_DIR", os.path.expanduser("~/.ash"))
        config_file = os.path.join(ash_dir, "site_model_list.txt")

        model_config = {}
        if not os.path.exists(config_file):
            return model_config

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                current_section = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        current_section = line[1:-1]
                        model_config[current_section] = {}
                    elif current_section and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        model_config[current_section][key] = value
        except Exception as e:
            print(f"Warning: Could not load model config: {e}")
        return model_config

    def _get_current_model_nickname(self) -> str:
        """Determine current model nickname from AI session config"""
        provider = self.ai.cfg.get("ASH_PROVIDER", "")
        model = self.ai.cfg.get("ASH_MODEL", "")

        for nickname, config in self.model_config.items():
            if config.get("ASH_PROVIDER") == provider and config.get("ASH_MODEL") == model:
                return nickname
        return None

    def switch_model(self, nickname: str):
        if nickname not in self.model_config:
            self.gui_callbacks["show_error_message"](f"Model {nickname} not found in configuration.")
            return

        config = self.model_config[nickname]

        if self.gui_callbacks["ask_confirmation"](f"Switch to model {nickname}?", "Confirm"):
            if self.ai.provider:
                self.ai.close_ai_session()

            for key, value in config.items():
                os.environ[key] = value
                self.ai.cfg[key] = value

            self.ai.open_ai_session()
            if not self.ai.provider:
                self.gui_callbacks["show_error_message"]("Failed to initialize new AI session.")
                return

            self.current_model_nickname = nickname
            self.gui_callbacks["update_title"](self.ai.cfg.get("ASH_PROVIDER", "unknown"), self.ai.cfg.get("ASH_MODEL", "unknown"))
            self.gui_callbacks["append_chat"]("system", f"Switched to model: {nickname}")
            self.gui_callbacks["update_status"]()
            # After switching models, refresh the loaded files list from the new session
            self._send_command_to_ai("list files", append_to_chat=False, disable_controls=False)
        else:
            self.gui_callbacks["revert_model_selection"]()

    def _send_command_to_ai(self, command: str, append_to_chat: bool = True, disable_controls: bool = True):
        """Helper to send a command directly to AI without UI input field.
        This is for internal file management commands that should be processed by eda_ai_assist.
        """
        if not self.ai.provider:
            self.gui_callbacks["show_error_message"]("AI session is not active for command.")
            return

        if append_to_chat:
            self.gui_callbacks["append_chat"]("user", command)
        if disable_controls:
            self.gui_callbacks["disable_controls"]()

        thread = threading.Thread(target=self._ai_request_thread, args=(command,), daemon=True)
        thread.start()

    def load_file(self, file_path: str):
        if file_path:
            # Check if file is already in AI's session_file_list by full path
            if file_path in self.ai.session_file_list:
                self.gui_callbacks["show_info_message"](f"{os.path.basename(file_path)} is already loaded.")
            else:
                self._send_command_to_ai(f"input file \"{file_path}\"")
                # After loading, explicitly ask AI to list files to refresh GUI
                self._send_command_to_ai("list files", append_to_chat=False, disable_controls=False)


    def unload_files(self, selected_full_paths: List[str]): # Changed parameter name
        if not selected_full_paths:
            return

        # No need for basename re-mapping, selected_full_paths are directly from the listbox
        for file_path in selected_full_paths:
            self._send_command_to_ai(f"delete file \"{file_path}\"")
        
        # After deleting, explicitly ask AI to list files to refresh GUI
        if selected_full_paths: # Only list files if something was actually deleted
            self._send_command_to_ai("list files", append_to_chat=False, disable_controls=False)


    def send_message(self, prompt: str):
        if not prompt:
            self.gui_callbacks["show_warning_message"]("Please enter a message.", "Empty Prompt")
            return

        if not self.ai.provider:
            self.gui_callbacks["show_error_message"]("AI session is not active.")
            return

        # Re-introduce logic to prepend loaded files to the prompt for AI context
        enhanced_prompt = ""
        if self.loaded_files: # self.loaded_files now mirrors self.ai.session_file_list via process_queue
            for file_path in self.loaded_files:
               # Use 'file' keyword and quote paths to handle spaces correctly
               enhanced_prompt += f"file \"{file_path}\"\n"
        enhanced_prompt += prompt
        
        # Original print statement (for debugging, can be removed in production)
        # print( enhanced_prompt ) 

        self.gui_callbacks["append_chat"]("user", prompt) # Display original user prompt
        self.gui_callbacks["clear_input_text"]()
        self.gui_callbacks["disable_controls"]()

        # The ai.session_file_list is managed by the eda_ai_assist module based on "input file" directives
        # and its own internal logic, so we do not reset it here.
        thread = threading.Thread(target=self._ai_request_thread, args=(enhanced_prompt,), daemon=True)
        thread.start()

    def _ai_request_thread(self, prompt: str):
        try:
            response = self.ai.ask_ai(prompt)
            warnings = self.ai.get_warnings()
            self.response_queue.put(("response", response, warnings))
        except Exception as e:
            self.response_queue.put(("error", str(e), None))
        finally:
            self.gui_callbacks["re_enable_controls"]()

    def process_queue(self):
        try:
            while True:
                msg_type, data, extra = self.response_queue.get_nowait()
                if msg_type == "response":
                    # Check if the response is a system message (e.g., from local file command in eda_ai_assist)
                    if data.startswith("[system]: "):
                        self.gui_callbacks["append_chat"]("system", data[len("[system]: "):])
                    else:
                        self.gui_callbacks["append_chat"]("assistant", data)
                    
                    if extra:
                        for warning in extra:
                            self.gui_callbacks["append_chat"]("system", f"WARNING: {warning}")
                    
                    # SYNCHRONIZE CORE'S LOADED_FILES (for GUI display and prompt construction)
                    # WITH AI'S SESSION_FILE_LIST (the true state of files tracked by AI)
                    # This captures files loaded explicitly via GUI and files discovered by AI in prompts
                    unique_ai_files = list(sorted(set(self.ai.session_file_list)))
                    if unique_ai_files != self.loaded_files: # Only update if there's a change
                        self.loaded_files = unique_ai_files
                        # Now call the GUI update, which reads from self.loaded_files
                        self.gui_callbacks["update_loaded_files_display"]()

                    self.gui_callbacks["update_status"]()
                    if not self.ai.provider:
                        self.gui_callbacks["append_chat"]("system", "AI session was closed due to token limits.")
                elif msg_type == "error":
                    self.gui_callbacks["append_chat"]("system", f"Error: {data}")
        except queue.Empty:
            pass

    def get_status_text(self) -> str:
        if self.ai.provider:
            tokens = self.ai.token_cnt_total
            files = len(self.ai.session_file_list)
            response_time = self.ai.last_response_time
            return f"Tokens: {tokens:,} | Files: {files} | Response Time: {response_time:.2f}s"
        else:
            return "Status: AI session not active"

    def get_about_text(self) -> str:
        return """
AshChat - AI-Enabled EDA Assistant GUI Frontend
Version 1.0

Author: Kevin M. Hubbard
Organization: Black Mesa Labs
Location: Sammamish, WA
Operational Since: February 8th, 2026

LICENSE
Licensed under GNU General Public License v3 or later.
See https://www.gnu.org/licenses/ for full text.

PROJECT
ash (eda_ai_assist) is a cross-platform tool for natural-language analysis
of EDA files using cloud-based AI providers.

Repository: https://github.com/blackmesalabs/eda_ai_assist

PROVIDERS SUPPORTED
- Google Gemini
- Microsoft Azure OpenAI
- Amazon Bedrock (Claude, Nova)
"""

    def get_help_text(self) -> str:
        return """
ASHCHAT - Quick Reference Guide

BASIC OPERATION
- Type your prompt in the input area at the bottom.
- Press Enter to submit your prompt (when in single-line mode).
- Press Ctrl+Enter to submit your prompt (works in both single-line and multi-line modes).
- Press Ctrl+D to toggle between normal (single-line) and expanded multi-line input modes.
- In multi-line mode, Enter inserts a newline. Use Ctrl+Enter or the "Send" button to submit.
- Shift+Enter always inserts a newline.
- Use the chat history area to review responses.

FILE OPERATIONS (File Menu)
- Load File: Add files to Loaded Files panel. Handled by AI "input file" command.
- Unload File: Remove files from Loaded Files panel. Handled by AI "delete file" command.
- Save Chat Transcript: Save entire conversation to text file.
- Save Selected Text: Save only highlighted text.

LOADED FILES PANEL
- Shows all currently loaded files at top of window.
- These files are managed by the Ash AI session.
- Use File > Unload File to remove files from this panel.

TEXT EDITING (Edit Menu)
- Copy: Copy selected text from chat history.
- Paste: Paste text from clipboard into input area.
- Select All: Select all text in chat history.
- Clear Chat: Erase all chat history (cannot be undone).

FORMATTING (Format Menu)
- Adjust font size from 8pt to 18pt for all GUI elements.
- Reset to Default: Return to system-detected default size.

AI MODEL SELECTION (AI Menu)
- Two-tier menu organized by provider and model nickname.
- Select different model to switch AI providers/models.
- Current session closes, new session opens with selected config.
- Loaded files persist when switching models.
- Window title updates to show new provider and model.
- Currently selected model shows as checked radio button.

SESSION MANAGEMENT (AI Menu)
- Flush Session: Clear history, unload files, reset tokens, start fresh.

KEYBOARD SHORTCUTS
Enter          Send message (single-line mode only)
Ctrl+Enter     Send message (both modes)
Ctrl+D         Toggle multi-line input mode
Shift+Enter    Insert newline (always)
Ctrl+C         Copy selected text (standard)
Ctrl+V         Paste (standard, use Edit menu)
"""

    def flush_session(self):
        if self.gui_callbacks["ask_confirmation"]("Flush AI session, clear chat, and unload all files?", "Confirm"):
            self.gui_callbacks["disable_controls"]()
            try:
                # First, ensure all cloud files are deleted via AI command
                self._send_command_to_ai("delete *", append_to_chat=False, disable_controls=False)
                
                # Close and re-open AI session
                self.ai.close_ai_session()
                self.gui_callbacks["clear_chat_display"]()
                self.loaded_files.clear() # Clear local cache
                self.ai.session_file_list.clear() # Clear eda_ai_assist's list explicitly
                self.gui_callbacks["update_loaded_files_display"]() # Force update to clear display

                self.ai.open_ai_session()
                if self.ai.provider:
                    self.gui_callbacks["append_chat"]("system", "AI session flushed. Starting new conversation.")
                    self.gui_callbacks["update_status"]()
                    # Refresh the file list after flushing - will be empty
                    self._send_command_to_ai("list files", append_to_chat=False, disable_controls=False)
                else:
                    self.gui_callbacks["show_error_message"]("Failed to restart AI session.")
            finally:
                self.gui_callbacks["re_enable_controls"]()

    def get_exit_message(self) -> str:
        cost_text = ""
        if self.ai.provider:
            model = self.ai.cfg.get("ASH_MODEL", "")
            ash_dir = self.ai.cfg.get("ASH_DIR", "")
            up = self.ai.token_cnt_upload
            down = self.ai.token_cnt_download
            cost_text = self.ai.ash_report_session_cost(model, ash_dir, up, down)

        # Append any warnings/cost from close_ai_session if it was called implicitly
        all_warnings = self.ai.get_warnings()
        if all_warnings:
            for w in all_warnings:
                if "Estimated total cost:" in w: # Look for cost info specifically
                    if cost_text: cost_text += "\n" + w
                    else: cost_text = w
                else: # Other warnings can be logged or ignored in exit message
                    pass # Current implementation prints warnings to console, not GUI exit msg

        msg = "Close AshChat and terminate AI session?"
        if cost_text:
            msg += "\n\n" + cost_text
        return msg

    def close_ai_session(self):
        if self.ai.provider:
            self.ai.close_ai_session()


# --- wxPython Frontend ---
if USE_WXPYTHON:
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

            # Callbacks for AshChatCore
            gui_callbacks = {
                "show_error_message": lambda msg: wx.MessageBox(msg, "Error", wx.OK | wx.ICON_ERROR),
                "show_warning_message": lambda msg, title: wx.MessageBox(msg, title, wx.OK | wx.ICON_WARNING),
                "show_info_message": lambda msg: wx.MessageBox(msg, "Info", wx.OK | wx.ICON_INFORMATION),
                "ask_confirmation": lambda msg, title: wx.MessageBox(msg, title, wx.YES_NO | wx.ICON_QUESTION) == wx.YES,
                "update_title": self._update_title,
                "append_chat": self._append_chat,
                "update_status": self._update_status,
                "update_loaded_files_display": self._update_loaded_files_display, # Pass function directly
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

            # Fix for Issue 1: Delay initial file list refresh until GUI is fully set up
            wx.CallAfter(self._initial_file_list_refresh)

        def _initial_file_list_refresh(self):
            # This will trigger an AI call which then updates core.loaded_files and the GUI
            self.core._send_command_to_ai("list files", append_to_chat=False, disable_controls=False)

        def _update_title(self, provider: str, model: str):
            self.SetTitle(f"AshChat - {provider}:{model}")

        def _revert_model_selection_wx(self):
            if self.core.current_model_nickname and self.core.current_model_nickname in self.model_radio_group:
                self.model_radio_group[self.core.current_model_nickname].Check(True)

        def _setup_menu(self):
            menubar = wx.MenuBar()

            # File menu
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

            # Edit menu
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

            # Format menu
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

            # AI menu (model selection and session management)
            ai_menu = wx.Menu()
            self.model_radio_group = {}
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
                        menu_item = provider_submenu.AppendRadioItem(wx.ID_ANY, nickname, f"Switch to {nickname} model")
                        self.Bind(wx.EVT_MENU, lambda evt, n=nickname: self.core.switch_model(n), menu_item)
                        self.model_radio_group[nickname] = menu_item

                    ai_menu.AppendSubMenu(provider_submenu, provider_name)

                if self.core.current_model_nickname and self.core.current_model_nickname in self.model_radio_group:
                    self.model_radio_group[self.core.current_model_nickname].Check(True)
            else:
                ai_menu.Append(wx.ID_ANY, "(No models configured)", "No AI models found", wx.ITEM_NORMAL).Enable(False)
            
            ai_menu.AppendSeparator()
            flush_session_menu_item = ai_menu.Append(wx.ID_ANY, "&Flush Session", "Clear history, unload files, reset tokens")
            self.Bind(wx.EVT_MENU, self._flush_session_wx, flush_session_menu_item)

            menubar.Append(ai_menu, "&AI")

            # Help menu
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

            files_panel = wx.Panel(self)
            # Updated label to reflect management by AI session
            files_box = wx.StaticBox(files_panel, label="Loaded Files (managed by Ash AI session)")
            files_box.SetFont(self.gui_font_bold)
            self.font_updatable_widgets.append(files_box)

            files_sizer = wx.StaticBoxSizer(files_box, wx.VERTICAL)

            self.loaded_files_listbox = wx.ListBox(
                files_panel,
                style=wx.LB_SINGLE # Use single selection for now, MultiChoiceDialog handles multiple
            )
            self.loaded_files_listbox.SetFont(self.mono_font)
            self.font_updatable_widgets.append(self.loaded_files_listbox)
            files_sizer.Add(self.loaded_files_listbox, 1, wx.EXPAND | wx.ALL, 5)
            files_panel.SetSizer(files_sizer)
            main_sizer.Add(files_panel, 0, wx.EXPAND | wx.ALL, 5)

            chat_panel = wx.Panel(self)
            chat_sizer = wx.BoxSizer(wx.VERTICAL)
            chat_label = wx.StaticText(chat_panel, label="Chat History")
            chat_label.SetFont(self.gui_font_bold)
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
            main_sizer.Add(chat_panel, 1, wx.EXPAND | wx.ALL, 5)

            self.chat_display.Bind(wx.EVT_CONTEXT_MENU, self._on_chat_display_context_menu)

            input_panel = wx.Panel(self)
            input_sizer = wx.BoxSizer(wx.VERTICAL)
            # Updated label text
            input_label = wx.StaticText(input_panel, label="Prompt (Enter to send, Ctrl+Enter to force send, Ctrl+D for multi-line)")
            input_label.SetFont(self.gui_font)
            self.font_updatable_widgets.append(input_label)
            input_sizer.Add(input_label, 0, wx.ALIGN_LEFT | wx.LEFT | wx.TOP, 5)

            self.input_text = wx.TextCtrl(
                input_panel,
                # Changed to word wrap, removed horizontal scroll
                style=wx.TE_MULTILINE | wx.TE_WORDWRAP | wx.VSCROLL 
            )
            self.input_text.SetMinSize((-1, 75))
            self.input_text.SetFont(self.mono_font)
            self.font_updatable_widgets.append(self.input_text)
            input_sizer.Add(self.input_text, 1, wx.EXPAND | wx.ALL, 5)
            self.input_text.Bind(wx.EVT_KEY_DOWN, self._on_input_char_wx)

            button_sizer = wx.BoxSizer(wx.HORIZONTAL)
            
            # Send button is dynamic and initially hidden
            self.send_button = wx.Button(input_panel, label="Send")
            self.send_button.SetBackgroundColour(wx.Colour("#4CAF50"))
            self.send_button.SetForegroundColour(wx.WHITE)
            self.send_button.SetFont(self.gui_font_bold)
            self.send_button.Bind(wx.EVT_BUTTON, self._send_message_wx)
            self.send_button.Hide() # Initially hidden
            self.font_updatable_widgets.append(self.send_button)
            button_sizer.Add(self.send_button, 0, wx.LEFT, 5)

            input_sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5)
            input_panel.SetSizer(input_sizer)
            main_sizer.Add(input_panel, 0, wx.EXPAND | wx.ALL, 5)

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

        # Modified to read from self.core.loaded_files (which is synchronized by process_queue)
        def _update_loaded_files_display(self):
            self.loaded_files_listbox.Clear()
            # Directly add full paths to the listbox for unambiguous display
            for file_path in sorted(self.core.loaded_files): # Sort for consistent display
                self.loaded_files_listbox.Append(file_path)
         

        def _on_input_char_wx(self, event):
            keycode = event.GetKeyCode()
            # Handle Ctrl+Enter (sends in both modes)
            if keycode == wx.WXK_RETURN and event.CmdDown():
                self._send_message_wx()
                return
            # Handle plain Enter
            if keycode == wx.WXK_RETURN:
                if event.ShiftDown():
                    # Shift+Enter always inserts a newline
                    event.Skip()
                    return
                else: # Plain Enter
                    if not self.input_multiline_expanded:
                        self._send_message_wx()
                        return # Consume the event
                    else:
                        # In multi-line mode, plain Enter inserts a newline
                        event.Skip()
                        return
            # Handle Ctrl+D for toggle multi-line
            if keycode == ord('D') and event.CmdDown():
                self._toggle_multiline_wx()
                return
            event.Skip() # For other keys

        def _send_message_wx(self, event=None):
            prompt = self.input_text.GetValue().strip()
            self.core.send_message(prompt)

        def _disable_controls(self):
            # Check if widgets exist before attempting to disable
            if hasattr(self, 'send_button') and self.send_button.IsShown(): 
                self.send_button.Disable()
            if hasattr(self, 'input_text'): self.input_text.Disable()

        def _re_enable_controls(self):
            # Check if widgets exist before attempting to enable
            if hasattr(self, 'send_button') and self.send_button.IsShown(): 
                wx.CallAfter(self.send_button.Enable)
            if hasattr(self, 'input_text'):
                wx.CallAfter(self.input_text.Enable)
                wx.CallAfter(self.input_text.SetFocus)

        def _process_queue_wx(self, event=None):
            self.core.process_queue()

        def _toggle_multiline_wx(self):
            if not self.input_multiline_expanded:
                self.input_text.SetMinSize((-1, 250)) # Made taller
                self.input_multiline_expanded = True
                self.send_button.Show() # Show Send button in multi-line mode
            else:
                self.input_text.SetMinSize((-1, 75))
                self.input_multiline_expanded = False
                self.send_button.Hide() # Hide Send button in single-line mode
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
                        widget.SetFont(self.gui_font_bold)
                    elif isinstance(widget, wx.StaticText) or isinstance(widget, wx.Button):
                        original_font = widget.GetFont()
                        if original_font.GetWeight() == wx.FONTWEIGHT_BOLD:
                            widget.SetFont(self.gui_font_bold)
                        else:
                            widget.SetFont(self.gui_font)
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
            if not self.core.loaded_files: # Use core's local list which is synchronized
                wx.MessageBox("No files to unload.", "Info", wx.OK | wx.ICON_INFORMATION)
                return

            # Display full paths for unambiguous selection
            current_display_paths = list(self.core.loaded_files)

            with wx.MultiChoiceDialog(
                self, "Select file(s) to unload:", "Unload File",
                current_display_paths # Display full paths
            ) as dialog:
                if dialog.ShowModal() == wx.ID_OK:
                    selections = dialog.GetSelections()
                    if selections:
                        # Map selected indices back to full paths
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
                        if wx.TheClipboard.IsOpened(): wx.TheClipboard.Close()
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
                self.queue_timer.Stop()
                self.Destroy()

    class AshChatApp(wx.App):
        def OnInit(self):
            frame = WxFrontend(None, title="AshChat")
            return True

    def main_gui():
        app = AshChatApp()
        app.MainLoop()

# --- Tkinter Frontend ---
else: # USE_WXPYTHON is False
    class TkFrontend:
        def __init__(self, root):
            self.root = root
            self.root.geometry("900x700")
            self.root.minsize(600, 400)

            self.default_font_size = 10 if sys.platform.startswith("win32") else 11
            self.default_gui_font_size = 10 if sys.platform.startswith("win32") else 11
            self.current_font_size = self.default_font_size
            self.current_gui_font_size = self.default_gui_font_size

            self.input_multiline_expanded = False
            self.ui_components = {"labels": [], "buttons": []}

            # Callbacks for AshChatCore
            gui_callbacks = {
                "show_error_message": lambda msg: messagebox.showerror("Error", msg),
                "show_warning_message": lambda msg, title: messagebox.showwarning(title, msg),
                "show_info_message": lambda msg: messagebox.showinfo("Info", msg),
                "ask_confirmation": lambda msg, title: messagebox.askyesno(title, msg),
                "update_title": self._update_title,
                "append_chat": self._append_chat,
                "update_status": self._update_status,
                "update_loaded_files_display": self._update_loaded_files_display,
                "clear_input_text": lambda: self.input_text.delete("1.0", tk.END),
                "disable_controls": self._disable_controls,
                "re_enable_controls": self._re_enable_controls,
                "clear_chat_display": lambda: self.chat_display.config(state=tk.NORMAL) or self.chat_display.delete("1.0", tk.END) or self.chat_display.config(state=tk.DISABLED),
                "revert_model_selection": self._revert_model_selection_tk,
            }
            self.core = AshChatCore(gui_callbacks)

            self.selected_model_var = tk.StringVar()
            self._setup_menu()
            self._setup_loaded_files_panel()
            self._setup_chat_display()
            self._setup_input_area()
            self._setup_status_bar()

            self._update_title(self.core.ai.cfg.get("ASH_PROVIDER", "unknown"), self.core.ai.cfg.get("ASH_MODEL", "unknown"))
            self.status_bar.config(text=self.core.get_status_text())

            # Start queue processing
            self.root.after(100, self._process_queue_tk)
            self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

            # Fix for Issue 1: Delay initial file list refresh until GUI is fully set up
            self.root.after(100, self._initial_file_list_refresh)


        def _initial_file_list_refresh(self):
            # This will trigger an AI call which then updates core.loaded_files and the GUI
            self.core._send_command_to_ai("list files", append_to_chat=False, disable_controls=False)


        def _update_title(self, provider: str, model: str):
            self.root.title(f"AshChat - {provider}:{model}")

        def _revert_model_selection_tk(self):
            if self.core.current_model_nickname:
                self.selected_model_var.set(self.core.current_model_nickname)

        def _setup_menu(self):
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)

            # File menu
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Load File", command=self._load_file_tk)
            file_menu.add_command(label="Unload File", command=self._unload_file_tk)
            file_menu.add_separator()
            file_menu.add_command(label="Save Chat Transcript", command=self._save_transcript_tk)
            file_menu.add_command(label="Save Selected Text", command=self._save_selected_tk)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self._on_exit)

            # Edit menu
            edit_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Edit", menu=edit_menu)
            edit_menu.add_command(label="Copy", command=self._copy_text_tk)
            edit_menu.add_command(label="Paste", command=self._paste_text_tk)
            edit_menu.add_command(label="Select All", command=self._select_all_tk)
            edit_menu.add_separator()
            edit_menu.add_command(label="Clear Chat", command=self._clear_chat_tk)

            # Format menu
            format_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Format", menu=format_menu)
            for size in range(8, 19):
                format_menu.add_command(label=f"Font Size {size}pt", command=lambda s=size: self._set_font_size(s))
            format_menu.add_separator()
            format_menu.add_command(label="Reset to Default", command=self._reset_font_size)

            # AI menu (model selection and session management)
            ai_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="AI", menu=ai_menu)

            if self.core.model_config:
                providers = {}
                for nickname, config in self.core.model_config.items():
                    provider = config.get("ASH_PROVIDER", "unknown")
                    if provider not in providers:
                        providers[provider] = []
                    providers[provider].append(nickname)

                for provider in sorted(providers.keys()):
                    provider_menu = tk.Menu(ai_menu, tearoff=0)
                    ai_menu.add_cascade(label=provider, menu=provider_menu)

                    for nickname in sorted(providers[provider]):
                        provider_menu.add_radiobutton(
                            label=nickname,
                            variable=self.selected_model_var,
                            value=nickname,
                            command=lambda n=nickname: self.core.switch_model(n)
                        )
                if self.core.current_model_nickname:
                    self.selected_model_var.set(self.core.current_model_nickname)
            else:
                ai_menu.add_command(label="(No models configured)", state=tk.DISABLED)

            ai_menu.add_separator()
            ai_menu.add_command(label="Flush Session", command=self._flush_session_tk)

            # Help menu
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="View Help", command=self._show_help_tk)
            help_menu.add_separator()
            help_menu.add_command(label="About", command=self._show_about_tk)

            self.root.bind("<Control-l>", lambda event: self._load_file_tk())
            self.root.bind("<Control-u>", lambda event: self._unload_file_tk())
            self.root.bind("<Control-s>", lambda event: self._save_selected_tk())
            self.root.bind("<Control-Shift-s>", lambda event: self._save_transcript_tk())
            self.root.bind("<Control-c>", lambda event: self._copy_text_tk())
            self.root.bind("<Control-v>", lambda event: self._paste_text_tk())
            self.root.bind("<Control-a>", lambda event: self._select_all_tk())


        def _setup_loaded_files_panel(self):
            files_frame = tk.LabelFrame(
                self.root,
                # Updated label to reflect management by AI session
                text="Loaded Files (managed by Ash AI session)",
                font=("Courier", self.current_gui_font_size, "bold")
            )
            files_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            self.ui_components["labels"].append(("files_frame_label", files_frame))

            files_scroll = tk.Scrollbar(files_frame)
            files_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            self.loaded_files_listbox = tk.Listbox(
                files_frame,
                height=3,
                font=("Courier", self.current_font_size),
                yscrollcommand=files_scroll.set,
                selectmode=tk.SINGLE
            )
            self.loaded_files_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            files_scroll.config(command=self.loaded_files_listbox.yview)

        def _setup_chat_display(self):
            chat_frame = tk.Frame(self.root)
            chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

            chat_label = tk.Label(
                chat_frame,
                text="Chat History",
                font=("Courier", self.current_gui_font_size, "bold")
            )
            chat_label.pack(anchor=tk.W)
            self.ui_components["labels"].append(("chat_label", chat_label))

            self.chat_display = scrolledtext.ScrolledText(
                chat_frame,
                height=20,
                width=100,
                state=tk.DISABLED,
                font=("Courier", self.current_font_size),
                bg="#f0f0f0",
                wrap=tk.WORD
            )
            self.chat_display.pack(fill=tk.BOTH, expand=True)

            self.chat_display.bind("<Key>", lambda e: "break")
            self.chat_display.bind("<Control-c>", lambda e: self._copy_text_tk())
            self.chat_display.bind("<Control-v>", lambda e: "break")
            self.chat_display.bind("<Delete>", lambda e: "break")
            self.chat_display.bind("<BackSpace>", lambda e: "break")

        def _setup_input_area(self):
            self.input_frame = tk.Frame(self.root)
            self.input_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=5, pady=5)

            input_label = tk.Label(
                self.input_frame,
                # Updated label text
                text="Prompt (Enter to send, Ctrl+Enter to force send, Ctrl+D for multi-line)",
                font=("Courier", self.current_gui_font_size)
            )
            input_label.pack(anchor=tk.W)
            self.ui_components["labels"].append(("input_label", input_label))

            self.input_text_frame = tk.Frame(self.input_frame)
            self.input_text_frame.pack(fill=tk.BOTH, expand=True)

            v_scroll = tk.Scrollbar(self.input_text_frame)
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            # Removed h_scroll for word wrap
            # h_scroll = tk.Scrollbar(self.input_text_frame, orient=tk.HORIZONTAL)
            # h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

            self.input_text = tk.Text(
                self.input_text_frame,
                height=5,
                width=100,
                font=("Courier", self.current_font_size),
                wrap=tk.WORD, # Changed to word wrap
                yscrollcommand=v_scroll.set # Removed xscrollcommand
            )
            self.input_text.pack(fill=tk.BOTH, expand=True)
            v_scroll.config(command=self.input_text.yview)
            # h_scroll.config(command=self.input_text.xview) # Removed h_scroll config

            self.input_text.bind("<Return>", self._on_return_key_tk) # Bind plain Enter
            self.input_text.bind("<Control-Return>", lambda e: self._send_message_tk()) # Ctrl+Enter always sends
            self.input_text.bind("<Shift-Return>", lambda e: "break") # Shift+Enter always inserts newline
            self.input_text.bind("<Control-d>", lambda e: self._toggle_multiline_tk())

            button_frame = tk.Frame(self.input_frame)
            button_frame.pack(fill=tk.X, pady=5)

            # Send button is dynamic and initially packed_forget
            self.send_button = tk.Button(
                button_frame,
                text="Send",
                command=self._send_message_tk,
                bg="#4CAF50",
                fg="white",
                font=("Courier", self.current_gui_font_size, "bold"),
                padx=20,
                pady=5
            )
            # Initially not packed; will be packed in _toggle_multiline_tk
            self.ui_components["buttons"].append(("send_button", self.send_button))

        def _setup_status_bar(self):
            self.status_bar = tk.Label(
                self.root,
                text=self.core.get_status_text(),
                font=("Courier", self.current_gui_font_size),
                bg="#e0e0e0",
                anchor=tk.W,
                padx=5,
                pady=3
            )
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
            self.ui_components["labels"].append(("status_bar", self.status_bar))

        def _update_status(self):
            self.status_bar.config(text=self.core.get_status_text())

        def _append_chat(self, role: str, message: str):
            self.chat_display.config(state=tk.NORMAL)
            if role == "user":
                self.chat_display.insert(tk.END, f"\n[You]: {message}\n", "user")
            elif role == "assistant":
                self.chat_display.insert(tk.END, f"\n[Ash]: {message}\n", "assistant")
            elif role == "system":
                self.chat_display.insert(tk.END, f"\n[System]: {message}\n", "system")
            self.chat_display.see(tk.END)
            self.chat_display.config(state=tk.DISABLED)

        # Modified to read from self.core.loaded_files (which is synchronized by process_queue)
        def _update_loaded_files_display(self):
            self.loaded_files_listbox.delete(0, tk.END)
            # Directly add full paths to the listbox for unambiguous display
            for file_path in sorted(self.core.loaded_files): # Sort for consistent display
                self.loaded_files_listbox.insert(tk.END, file_path)


        def _on_return_key_tk(self, event):
            # This handler is only for plain Enter (not Shift+Enter or Ctrl+Enter)
            if not self.input_multiline_expanded:
                self._send_message_tk()
                return "break" # Consume event, prevent default newline
            else:
                # In multi-line mode, plain Enter inserts a newline
                return # Allow default behavior (insert newline)


        def _send_message_tk(self):
            prompt = self.input_text.get("1.0", tk.END).strip()
            self.core.send_message(prompt)

        def _disable_controls(self):
            # Check if widgets exist before attempting to disable
            if hasattr(self, 'send_button') and self.send_button.winfo_ismapped(): 
                self.send_button.config(state=tk.DISABLED)
            if hasattr(self, 'input_text'): self.input_text.config(state=tk.DISABLED)

        def _re_enable_controls(self):
            # Check if widgets exist before attempting to enable
            if hasattr(self, 'send_button') and self.send_button.winfo_ismapped(): 
                self.send_button.config(state=tk.NORMAL)
            if hasattr(self, 'input_text'):
                self.input_text.config(state=tk.NORMAL)
                self.input_text.focus()

        def _process_queue_tk(self):
            self.core.process_queue()
            self.root.after(100, self._process_queue_tk)

        def _toggle_multiline_tk(self):
            self.input_frame.update_idletasks()
            if not self.input_multiline_expanded:
                self.input_text.config(height=20) # Made taller
                self.input_multiline_expanded = True
                self.send_button.pack(side=tk.RIGHT, padx=5) # Show Send button
            else:
                self.input_text.config(height=5)
                self.input_multiline_expanded = False
                self.send_button.pack_forget() # Hide Send button

        def _set_font_size(self, size: int):
            if 8 <= size <= 18:
                self.current_font_size = size
                self.current_gui_font_size = size

                self.chat_display.config(font=("Courier", size))
                self.input_text.config(font=("Courier", size))
                self.loaded_files_listbox.config(font=("Courier", size))

                for component_type, components in self.ui_components.items():
                    if component_type == "labels":
                        for label_name, label_widget in components:
                            # Apply bold only to specific labels that should be bold
                            if label_name in ["files_frame_label", "chat_label"]:
                                label_widget.config(font=("Courier", size, "bold"))
                            else:
                                label_widget.config(font=("Courier", size))
                    elif component_type == "buttons":
                        for button_name, button_widget in components:
                            button_widget.config(font=("Courier", size, "bold"))

        def _reset_font_size(self):
            self.current_font_size = self.default_font_size
            self.current_gui_font_size = self.default_gui_font_size
            self._set_font_size(self.default_font_size)

        def _load_file_tk(self):
            file_path = filedialog.askopenfilename(
                title="Select a file to load",
                filetypes=[("All files", "*.*"), ("Text files", "*.txt"), ("Verilog", "*.v"), ("VCD", "*.vcd")]
            )
            self.core.load_file(file_path)

        def _unload_file_tk(self):
            if not self.core.loaded_files: # Use core's local list which is synchronized
                messagebox.showinfo("Info", "No files to unload.")
                return

            # Display full paths for unambiguous selection
            current_display_paths = list(self.core.loaded_files)

            unload_window = tk.Toplevel(self.root)
            unload_window.title("Unload File")
            # Dynamically adjust height based on number of files
            num_files = len(current_display_paths)
            window_height = max(200, 100 + num_files * 25) # Base + 25px per file
            unload_window.geometry(f"400x{window_height}")

            title_label = tk.Label(
                unload_window,
                text="Select file(s) to unload:",
                font=("Courier", self.current_gui_font_size)
            )
            title_label.pack(pady=10)

            # Use a listbox for selection
            selection_listbox = tk.Listbox(
                unload_window,
                selectmode=tk.MULTIPLE,
                height=min(10, num_files), # Max 10 visible, or num_files if less
                font=("Courier", self.current_gui_font_size)
            )
            selection_listbox.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

            for full_path in current_display_paths:
                selection_listbox.insert(tk.END, full_path) # Insert full paths

            def confirm_unload():
                selected_indices = selection_listbox.curselection()
                selected_full_paths_for_core = [current_display_paths[i] for i in selected_indices]
                self.core.unload_files(selected_full_paths_for_core)
                unload_window.destroy()

            confirm_button = tk.Button(
                unload_window,
                text="Unload Selected",
                command=confirm_unload,
                font=("Courier", self.current_gui_font_size),
                padx=20,
                pady=5
            )
            confirm_button.pack(pady=10)


        def _save_transcript_tk(self):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    self.chat_display.config(state=tk.NORMAL) # Temporarily enable to get value
                    content = self.chat_display.get("1.0", tk.END)
                    self.chat_display.config(state=tk.DISABLED)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    messagebox.showinfo("Success", f"Chat transcript saved to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save transcript: {e}")

        def _save_selected_tk(self):
            try:
                selected = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
                if not selected:
                    messagebox.showwarning("No Selection", "Please select text to save.")
                    return
            except tk.TclError:
                messagebox.showwarning("No Selection", "Please select text to save.")
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(selected)
                    messagebox.showinfo("Success", f"Selected text saved to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save selected text: {e}")

        def _copy_text_tk(self):
            try:
                selected = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
                if selected:
                    self.root.clipboard_clear()
                    self.root.clipboard_append(selected)
                else:
                    messagebox.showwarning("No Selection", "Please select text to copy.")
            except tk.TclError:
                messagebox.showwarning("No Selection", "Please select text to copy.")

        def _paste_text_tk(self):
            try:
                text = self.root.clipboard_get()
                self.input_text.insert(tk.INSERT, text)
            except tk.TclError:
                messagebox.showerror("Error", "Failed to paste from clipboard.")

        def _select_all_tk(self):
            self.chat_display.tag_add(tk.SEL, "1.0", tk.END)
            self.chat_display.mark_set(tk.INSERT, "1.0")
            self.chat_display.see(tk.INSERT)

        def _clear_chat_tk(self):
            if messagebox.askyesno("Confirm", "Clear all chat history?"):
                self.chat_display.config(state=tk.NORMAL)
                self.chat_display.delete("1.0", tk.END)
                self.chat_display.config(state=tk.DISABLED)

        def _flush_session_tk(self):
            self.core.flush_session()

        def _show_scrollable_dialog_tk(self, title: str, text: str, width: int = 80, height: int = 30):
            dialog = tk.Toplevel(self.root)
            dialog.title(title)
            dialog.geometry(f"{width*8}x{height*14}")

            text_area = scrolledtext.ScrolledText(
                dialog, width=width, height=height, font=("Courier", self.current_gui_font_size),
                wrap=tk.WORD, state=tk.NORMAL
            )
            text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            text_area.insert("1.0", text)
            text_area.config(state=tk.DISABLED)

            close_button = tk.Button(
                dialog, text="Close", command=dialog.destroy,
                font=("Courier", self.current_gui_font_size), padx=20, pady=5
            )
            close_button.pack(pady=5)

        def _show_help_tk(self):
            self._show_scrollable_dialog_tk("Help - AshChat", self.core.get_help_text(), width=90, height=35)

        def _show_about_tk(self):
            self._show_scrollable_dialog_tk("About AshChat", self.core.get_about_text(), width=70, height=18)

        def _on_exit(self):
            msg = self.core.get_exit_message()
            if messagebox.askyesno("Exit", msg):
                self.core.close_ai_session()
                self.root.quit()

    def main_gui():
        root = tk.Tk()
        app = TkFrontend(root)
        root.mainloop()

# --- Main Entry Point ---
if __name__ == "__main__":
    if USE_WXPYTHON:
        main_gui()
    else:
        main_gui()
