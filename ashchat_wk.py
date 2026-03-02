import wx
import sys
import os
import threading
import queue
from typing import List, Dict

# Import the eda_ai_assist API
try:
    from eda_ai_assist import api_eda_ai_assist
except ImportError:
    print("Error: Could not import eda_ai_assist module. Ensure eda_ai_assist.py is in the same directory.")
    sys.exit(1)


class AshChatFrame(wx.Frame):
    def __init__(self, parent, title):
        super(AshChatFrame, self).__init__(parent, title=title, size=(900, 700))

        self.SetMinSize((600, 400))

        # Determine default font size based on OS
        if sys.platform.startswith("win32"):
            self.default_font_size = 10
            self.default_gui_font_size = 10
        else:
            self.default_font_size = 11
            self.default_gui_font_size = 11
        self.current_font_size = self.default_font_size
        self.current_gui_font_size = self.default_gui_font_size

        # Create initial font objects for all widgets
        self.mono_font = wx.Font(self.current_font_size, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.mono_font_bold = wx.Font(self.current_font_size, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.gui_font = wx.Font(self.current_gui_font_size, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.gui_font_bold = wx.Font(self.current_gui_font_size, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)

        # List to keep track of widgets for font updates
        self.font_updatable_widgets = []

        # Load model list from site_model_list.txt
        self.model_config = self._load_model_config()
        self.current_model_nickname = None

        # AI session
        self.ai = api_eda_ai_assist()
        self.ai.open_ai_session()
        if not self.ai.provider:
            wx.MessageBox("Failed to initialize AI session.", "Error", wx.OK | wx.ICON_ERROR)
            sys.exit(1)

        # Detect current model
        self.current_model_nickname = self._get_current_model_nickname()

        # Set window title with provider and model
        provider = self.ai.cfg.get("ASH_PROVIDER", "unknown")
        model = self.ai.cfg.get("ASH_MODEL", "unknown")
        self.SetTitle(f"AshChat - {provider}:{model}")

        # Queue for thread-safe message passing (wx.CallAfter will be used to process)
        self.response_queue = queue.Queue()

        # Loaded files list (automatically included in prompts)
        self.loaded_files: List[str] = []

        # Track input frame for dynamic resizing
        self.input_multiline_expanded = False

        # Setup GUI elements and layout
        self._setup_menu()
        self._setup_panels()

        # Set initial status bar text
        self.status_bar.SetStatusText(self._get_initial_status())

        # Timer to poll the response queue
        self.queue_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._process_queue, self.queue_timer)
        self.queue_timer.Start(100) # Poll every 100 milliseconds

        self.Centre()
        self.Show(True)

        # Bind the close event to our custom handler
        self.Bind(wx.EVT_CLOSE, self._on_exit)

        # Update font menu selection to show current size checked
        self._update_font_menu_selection()

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

    def _setup_menu(self):
        menubar = wx.MenuBar()

        # File menu
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_OPEN, "&Load File\tCtrl+L", "Load a file for analysis")
        self.Bind(wx.EVT_MENU, self._load_file, id=wx.ID_OPEN)
        file_menu.Append(wx.ID_DELETE, "&Unload File\tCtrl+U", "Unload selected file(s)")
        self.Bind(wx.EVT_MENU, self._unload_file, id=wx.ID_DELETE)
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_SAVEAS, "&Save Chat Transcript\tCtrl+Shift+S", "Save the entire chat history")
        self.Bind(wx.EVT_MENU, self._save_transcript, id=wx.ID_SAVEAS)
        file_menu.Append(wx.ID_SAVE, "Save &Selected Text\tCtrl+S", "Save selected text from chat history")
        self.Bind(wx.EVT_MENU, self._save_selected, id=wx.ID_SAVE)
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, "E&xit", "Terminate AshChat")
        self.Bind(wx.EVT_MENU, self._on_exit, id=wx.ID_EXIT)
        menubar.Append(file_menu, "&File")

        # Edit menu
        edit_menu = wx.Menu()
        edit_menu.Append(wx.ID_COPY, "&Copy\tCtrl+C", "Copy selected text")
        self.Bind(wx.EVT_MENU, self._copy_text, id=wx.ID_COPY)
        edit_menu.Append(wx.ID_PASTE, "&Paste\tCtrl+V", "Paste text into input area")
        self.Bind(wx.EVT_MENU, self._paste_text, id=wx.ID_PASTE)
        edit_menu.Append(wx.ID_SELECTALL, "Select &All\tCtrl+A", "Select all text in chat history")
        self.Bind(wx.EVT_MENU, self._select_all, id=wx.ID_SELECTALL)
        edit_menu.AppendSeparator()
        edit_menu.Append(wx.ID_CLEAR, "&Clear Chat", "Clear all chat history")
        self.Bind(wx.EVT_MENU, self._clear_chat, id=wx.ID_CLEAR)
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

        # AI menu (model selection)
        ai_menu = wx.Menu()
        self.model_radio_group = {}
        if self.model_config:
            providers = {}
            for nickname, config in self.model_config.items():
                provider = config.get("ASH_PROVIDER", "unknown")
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(nickname)

            for provider_name in sorted(providers.keys()):
                provider_submenu = wx.Menu()
                for nickname in sorted(providers[provider_name]):
                    menu_item = provider_submenu.AppendRadioItem(wx.ID_ANY, nickname, f"Switch to {nickname} model")
                    self.Bind(wx.EVT_MENU, lambda evt, n=nickname: self._switch_model(n), menu_item)
                    self.model_radio_group[nickname] = menu_item

                ai_menu.AppendSubMenu(provider_submenu, provider_name)

            if self.current_model_nickname and self.current_model_nickname in self.model_radio_group:
                self.model_radio_group[self.current_model_nickname].Check(True)
        else:
            ai_menu.Append(wx.ID_ANY, "(No models configured)", "No AI models found", wx.ITEM_NORMAL).Enable(False)
        menubar.Append(ai_menu, "&AI")

        # Help menu
        help_menu = wx.Menu()
        help_menu.Append(wx.ID_HELP, "View &Help\tF1", "Show quick reference guide")
        self.Bind(wx.EVT_MENU, self._show_help, id=wx.ID_HELP)
        help_menu.AppendSeparator()
        help_menu.Append(wx.ID_ABOUT, "&About AshChat", "Show information about AshChat")
        self.Bind(wx.EVT_MENU, self._show_about, id=wx.ID_ABOUT)
        menubar.Append(help_menu, "&Help")

        self.SetMenuBar(menubar)

    def _update_font_menu_selection(self):
        for size, item in self.font_size_menu_items.items():
            item.Check(False)
        if self.current_font_size in self.font_size_menu_items:
            self.font_size_menu_items[self.current_font_size].Check(True)

    def _switch_model(self, nickname: str):
        """Switch to a different AI model"""
        if nickname not in self.model_config:
            wx.MessageBox(f"Model {nickname} not found in configuration.", "Error", wx.OK | wx.ICON_ERROR)
            return

        config = self.model_config[nickname]

        if wx.MessageBox(f"Switch to model {nickname}?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            if self.ai.provider:
                self.ai.close_ai_session()

            for key, value in config.items():
                os.environ[key] = value
                self.ai.cfg[key] = value

            self.ai.open_ai_session()
            if not self.ai.provider:
                wx.MessageBox("Failed to initialize new AI session.", "Error", wx.OK | wx.ICON_ERROR)
                return

            self.current_model_nickname = nickname

            provider = self.ai.cfg.get("ASH_PROVIDER", "unknown")
            model = self.ai.cfg.get("ASH_MODEL", "unknown")
            self.SetTitle(f"AshChat - {provider}:{model}")

            self._append_chat("system", f"Switched to model: {nickname}")
            self._update_status()
        else:
            if self.current_model_nickname and self.current_model_nickname in self.model_radio_group:
                self.model_radio_group[self.current_model_nickname].Check(True)

    def _setup_panels(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        files_panel = wx.Panel(self)
        files_box = wx.StaticBox(files_panel, label="Loaded Files (auto-included in prompts)")
        files_box.SetFont(self.gui_font_bold)
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
        main_sizer.Add(files_panel, 0, wx.EXPAND | wx.ALL, 5)

        chat_panel = wx.Panel(self)
        chat_sizer = wx.BoxSizer(wx.VERTICAL)
        chat_label = wx.StaticText(chat_panel, label="Chat History")
        chat_label.SetFont(self.gui_font_bold)
        self.font_updatable_widgets.append(chat_label)
        chat_sizer.Add(chat_label, 0, wx.ALIGN_LEFT | wx.LEFT | wx.TOP, 5)

        # Changed to word wrap and removed horizontal scrollbar
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

        # Bind context menu for chat_display to enable Copy via right-click
        self.chat_display.Bind(wx.EVT_CONTEXT_MENU, self._on_chat_display_context_menu)


        input_panel = wx.Panel(self)
        input_sizer = wx.BoxSizer(wx.VERTICAL)
        input_label = wx.StaticText(input_panel, label="Prompt (Ctrl+Enter to send, Ctrl+D for multi-line)")
        input_label.SetFont(self.gui_font)
        self.font_updatable_widgets.append(input_label)
        input_sizer.Add(input_label, 0, wx.ALIGN_LEFT | wx.LEFT | wx.TOP, 5)

        self.input_text = wx.TextCtrl(
            input_panel,
            style=wx.TE_MULTILINE | wx.HSCROLL | wx.VSCROLL
        )
        self.input_text.SetMinSize((-1, 75))
        self.input_text.SetFont(self.mono_font)
        self.font_updatable_widgets.append(self.input_text)
        input_sizer.Add(self.input_text, 1, wx.EXPAND | wx.ALL, 5)

        # Corrected binding: Use wx.EVT_KEY_DOWN for keyboard events
        self.input_text.Bind(wx.EVT_KEY_DOWN, self._on_input_char)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.flush_button = wx.Button(input_panel, label="Flush Session")
        self.flush_button.SetBackgroundColour(wx.Colour("#FF9800"))
        self.flush_button.SetForegroundColour(wx.WHITE)
        self.flush_button.SetFont(self.gui_font_bold)
        self.flush_button.Bind(wx.EVT_BUTTON, self._flush_session)
        self.font_updatable_widgets.append(self.flush_button)
        button_sizer.Add(self.flush_button, 0, wx.RIGHT, 5)

        self.send_button = wx.Button(input_panel, label="Send")
        self.send_button.SetBackgroundColour(wx.Colour("#4CAF50"))
        self.send_button.SetForegroundColour(wx.WHITE)
        self.send_button.SetFont(self.gui_font_bold)
        self.send_button.Bind(wx.EVT_BUTTON, self._send_message)
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

    def _get_initial_status(self) -> str:
        return "Tokens: 0 | Files: 0 | Response Time: N/A"

    def _update_status(self):
        if self.ai.provider:
            tokens = self.ai.token_cnt_total
            files = len(self.ai.session_file_list)
            response_time = self.ai.last_response_time
            status_text = f"Tokens: {tokens:,} | Files: {files} | Response Time: {response_time:.2f}s"
            self.status_bar.SetStatusText(status_text)
        else:
            status_text = "Status: AI session not active"
            self.status_bar.SetStatusText(status_text)

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
        for file_path in self.loaded_files:
            basename = os.path.basename(file_path)
            self.loaded_files_listbox.Append(basename)

    def _on_input_char(self, event):
        keycode = event.GetKeyCode()
        # Ctrl+Enter to send message
        if keycode == wx.WXK_RETURN and event.CmdDown(): # CmdDown for Ctrl on Windows/Linux, Command on Mac
            self._send_message()
            return
        # Ctrl+D to toggle multi-line
        if keycode == ord('D') and event.CmdDown():
            self._toggle_multiline()
            return
        event.Skip() # Important to allow other default TextCtrl events to be processed

    def _send_message(self, event=None):
        prompt = self.input_text.GetValue().strip()
        if not prompt:
            wx.MessageBox("Please enter a message.", "Empty Prompt", wx.OK | wx.ICON_WARNING)
            return

        if not self.ai.provider:
            wx.MessageBox("AI session is not active.", "Error", wx.OK | wx.ICON_ERROR)
            return

        if self.loaded_files:
            file_list_str = ", ".join(os.path.basename(f) for f in self.loaded_files)
            enhanced_prompt = f"Here are loaded files: {file_list_str}.\n\n{prompt}"
        else:
            enhanced_prompt = prompt

        self._append_chat("user", prompt)
        self.input_text.Clear()
        # Disable controls while AI is processing
        self.send_button.Disable()
        self.flush_button.Disable()
        self.input_text.Disable()

        thread = threading.Thread(target=self._ai_request_thread, args=(enhanced_prompt,), daemon=True)
        thread.start()

    def _ai_request_thread(self, prompt: str):
        try:
            response = self.ai.ask_ai(prompt)
            warnings = self.ai.get_warnings()
            wx.CallAfter(self.response_queue.put, ("response", response, warnings))
        except Exception as e:
            wx.CallAfter(self.response_queue.put, ("error", str(e), None))
        finally:
            # Always re-enable controls after the thread completes, regardless of success or failure
            wx.CallAfter(self._re_enable_controls)

    def _re_enable_controls(self):
        """Helper to re-enable GUI controls after a background operation."""
        self.send_button.Enable()
        self.flush_button.Enable()
        self.input_text.Enable()
        self.input_text.SetFocus()

    def _process_queue(self, event=None):
        # This is called by the timer to process items from the queue
        try:
            while True:
                msg_type, data, extra = self.response_queue.get_nowait()
                if msg_type == "response":
                    self._append_chat("assistant", data)
                    if extra:
                        for warning in extra:
                            self._append_chat("system", f"WARNING: {warning}")
                    self._update_status()
                    if not self.ai.provider:
                        self._append_chat("system", "AI session was closed due to token limits.")
                elif msg_type == "error":
                    self._append_chat("system", f"Error: {data}")
        except queue.Empty:
            pass
        # Controls are now re-enabled by _re_enable_controls from the thread's finally block,
        # so no re-enablement logic needed here.


    def _toggle_multiline(self):
        if not self.input_multiline_expanded:
            self.input_text.SetMinSize((-1, 200)) # Approximate height for 15 lines
            self.input_multiline_expanded = True
        else:
            self.input_text.SetMinSize((-1, 75)) # Approximate height for 5 lines
            self.input_multiline_expanded = False
        self.Layout() # Recalculate layout to apply new min size

    def _set_font_size(self, size: int):
        if 8 <= size <= 18:
            self.current_font_size = size
            self.current_gui_font_size = size

            # Update the font objects
            self.mono_font.SetPointSize(size)
            self.mono_font_bold.SetPointSize(size)
            self.gui_font.SetPointSize(size)
            self.gui_font_bold.SetPointSize(size)

            # Apply updated fonts to all tracked widgets
            for widget in self.font_updatable_widgets:
                if isinstance(widget, wx.StaticBox): # Labels of StaticBoxSizer
                    widget.SetFont(self.gui_font_bold)
                elif isinstance(widget, wx.StaticText) or isinstance(widget, wx.Button):
                    # Check if the original font was bold to retain styling
                    original_font = widget.GetFont()
                    if original_font.GetWeight() == wx.FONTWEIGHT_BOLD:
                        widget.SetFont(self.gui_font_bold)
                    else:
                        widget.SetFont(self.gui_font)
                elif isinstance(widget, wx.TextCtrl) or isinstance(widget, wx.ListBox):
                    widget.SetFont(self.mono_font)
                elif isinstance(widget, wx.StatusBar):
                    # Status bar font often needs special handling
                    sb_font = widget.GetFont()
                    sb_font.SetPointSize(size)
                    widget.SetFont(sb_font)

            # Re-apply font to menu bar. This might not change menu item fonts directly on all platforms.
            self.GetMenuBar().SetFont(self.gui_font)
            # More granular control over menu item fonts would require iterating through menus/items.

            self.Layout() # Recalculate layout after font changes
            self._update_font_menu_selection() # Update the checked item in the Format menu

    def _reset_font_size(self, event=None):
        self.current_font_size = self.default_font_size
        self.current_gui_font_size = self.default_gui_font_size
        self._set_font_size(self.default_font_size)

    def _load_file(self, event=None):
        with wx.FileDialog(
            self,
            "Select a file to load",
            wildcard="All files (*.*)|*.*|Text files (*.txt)|*.txt|Verilog (*.v)|*.v|VCD (*.vcd)|*.vcd",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return # User cancelled

            file_path = fileDialog.GetPath()
            if file_path:
                if file_path not in self.loaded_files:
                    self.loaded_files.append(file_path)
                    if file_path not in self.ai.session_file_list:
                        self.ai.session_file_list.append(file_path)
                    self._update_loaded_files_display()
                    self._append_chat("system", f"Loaded file: {os.path.basename(file_path)}")
                    self._update_status()
                else:
                    wx.MessageBox(f"{os.path.basename(file_path)} is already loaded.", "Info", wx.OK | wx.ICON_INFORMATION)

    def _unload_file(self, event=None):
        if not self.loaded_files:
            wx.MessageBox("No files to unload.", "Info", wx.OK | wx.ICON_INFORMATION)
            return

        # wx.MultiChoiceDialog allows selecting multiple items from a list
        with wx.MultiChoiceDialog(
            self,
            "Select file(s) to unload:",
            "Unload File",
            [os.path.basename(f) for f in self.loaded_files]
        ) as dialog:
            if dialog.ShowModal() == wx.ID_OK:
                selections = dialog.GetSelections()
                if selections:
                    # Get the actual file paths corresponding to selected indices
                    selected_file_paths = [self.loaded_files[i] for i in selections]

                    for file_path in selected_file_paths:
                        self.loaded_files.remove(file_path)
                        if file_path in self.ai.session_file_list:
                            self.ai.session_file_list.remove(file_path)
                    self._update_loaded_files_display()
                    self._append_chat("system", f"Unloaded {len(selected_file_paths)} file(s).")
                    self._update_status()

    def _save_transcript(self, event=None):
        with wx.FileDialog(
            self,
            "Save Chat Transcript",
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

    def _save_selected(self, event=None):
        # Use GetStringSelection() for wx.TextCtrl
        selected_text = self.chat_display.GetStringSelection()
        if not selected_text:
            wx.MessageBox("Please select text to save.", "No Selection", wx.OK | wx.ICON_WARNING)
            return

        with wx.FileDialog(
            self,
            "Save Selected Text",
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

    def _copy_text(self, event=None):
        selected_text = self.chat_display.GetStringSelection() # Corrected: Use GetStringSelection()
        if selected_text:
            if wx.TheClipboard.Open():
                try:
                    # Set the text on the clipboard
                    wx.TheClipboard.SetData(wx.TextDataObject(selected_text))
                    wx.TheClipboard.Close()
                    # Removed the wx.MessageBox confirmation here
                except Exception as e:
                    # In case of an error while setting data, attempt to close clipboard if it was opened.
                    if wx.TheClipboard.IsOpened():
                        wx.TheClipboard.Close()
                    wx.MessageBox(f"Failed to copy to clipboard: {e}", "Copy Error", wx.OK | wx.ICON_ERROR)
            else:
                wx.MessageBox("Failed to open clipboard.", "Clipboard Error", wx.OK | wx.ICON_ERROR)
        else:
            wx.MessageBox("Please select text in the Chat History to copy.", "No Selection", wx.OK | wx.ICON_WARNING)

    def _on_chat_display_context_menu(self, event):
        """Handler for the chat display's context menu."""
        menu = wx.Menu()
        copy_item = menu.Append(wx.ID_COPY, "Copy Selected Text")
        # Enable or disable copy item based on selection using GetStringSelection()
        copy_item.Enable(bool(self.chat_display.GetStringSelection())) # Corrected: Use GetStringSelection()
        self.Bind(wx.EVT_MENU, self._copy_text, copy_item)
        self.PopupMenu(menu)
        menu.Destroy()

    def _paste_text(self, event=None):
        if wx.TheClipboard.Open():
            if wx.TheClipboard.IsSupported(wx.DataFormat(wx.DF_TEXT)):
                tdo = wx.TextDataObject()
                wx.TheClipboard.GetData(tdo)
                text = tdo.GetText()
                self.input_text.WriteText(text) # WriteText inserts text at current cursor position
            else:
                wx.MessageBox("Clipboard does not contain text.", "Error", wx.OK | wx.ICON_WARNING)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Failed to open clipboard.", "Error", wx.OK | wx.ICON_ERROR)

    def _select_all(self, event=None):
        self.chat_display.SetSelection(-1, -1) # Selects all text in wx.TextCtrl

    def _clear_chat(self, event=None):
        if wx.MessageBox("Clear all chat history?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            self.chat_display.Clear()
            # self.chat_history.clear() # If this was actually used for state tracking, clear it.

    def _flush_session(self, event=None):
        if wx.MessageBox("Flush AI session, clear chat, and unload all files?", "Confirm", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            # Disable controls immediately before starting flush operations
            self.send_button.Disable()
            self.flush_button.Disable()
            self.input_text.Disable()

            try:
                # Close AI session
                self.ai.close_ai_session()

                # Clear chat display
                self.chat_display.Clear()

                # Clear loaded files
                self.loaded_files.clear()
                self.ai.session_file_list.clear()
                self._update_loaded_files_display()

                # Restart AI session
                self.ai.open_ai_session()
                if self.ai.provider:
                    self._append_chat("system", "AI session flushed. Starting new conversation.")
                    self._update_status()
                else:
                    wx.MessageBox("Failed to restart AI session.", "Error", wx.OK | wx.ICON_ERROR)
            finally:
                # Ensure controls are re-enabled after flush completes
                self._re_enable_controls()

    def _show_scrollable_dialog(self, title: str, text: str, width: int = 80, height: int = 30):
        # Using wx.Dialog for pop-up windows
        dialog = wx.Dialog(self, title=title, size=(width * 8, height * 14))
        dialog_sizer = wx.BoxSizer(wx.VERTICAL)

        text_ctrl = wx.TextCtrl(
            dialog,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.VSCROLL | wx.TE_WORDWRAP, # Added word wrap here too for consistency
            size=(width * 8 - 20, height * 14 - 100) # Adjust size to fit in dialog
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

    def _show_help(self, event=None):
        help_text = """
ASHCHAT - Quick Reference Guide

BASIC OPERATION
- Type your prompt in the input area at the bottom.
- Press Ctrl+Enter or click Send to submit.
- Press Ctrl+D to toggle between normal and expanded multi-line input modes.
- Use the chat history area to review responses.

FILE OPERATIONS (File Menu)
- Load File: Add files to Loaded Files panel. Auto-included in all prompts.
- Unload File: Remove files from Loaded Files panel.
- Save Chat Transcript: Save entire conversation to text file.
- Save Selected Text: Save only highlighted text.

LOADED FILES PANEL
- Shows all currently loaded files at top of window.
- Automatically referenced in every prompt sent.
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

SESSION MANAGEMENT
- Flush Session: Clear history, unload files, reset tokens, start fresh.
- Exit: Close GUI and terminate AI session.

STATUS BAR
- Shows token usage, loaded file count, and last response time.
- Window title displays provider and model.

KEYBOARD SHORTCUTS
Ctrl+Enter     Send message
Ctrl+D         Toggle multi-line input mode
Ctrl+C         Copy selected text (standard)
Ctrl+V         Paste (standard, use Edit menu)
"""
        self._show_scrollable_dialog("Help - AshChat", help_text, width=90, height=35)

    def _show_about(self, event=None):
        about_text = """
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
        self._show_scrollable_dialog("About AshChat", about_text, width=70, height=18)

    def _on_exit(self, event=None):
        cost_text = ""
        if self.ai.provider:
            model = self.ai.cfg.get("ASH_MODEL", "")
            ash_dir = self.ai.cfg.get("ASH_DIR", "")
            up = self.ai.token_cnt_upload
            down = self.ai.token_cnt_download
            cost_text = self.ai.ash_report_session_cost(model, ash_dir, up, down)

        msg = "Close AshChat and terminate AI session?"
        if cost_text:
            msg += "\n\n" + cost_text

        if wx.MessageBox(msg, "Exit", wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            if self.ai.provider:
                self.ai.close_ai_session()
            self.queue_timer.Stop()
            self.Destroy()


class AshChatApp(wx.App):
    def OnInit(self):
        frame = AshChatFrame(None, title="AshChat")
        return True

def main():
    app = AshChatApp()
    app.MainLoop()


if __name__ == "__main__":
    main()
