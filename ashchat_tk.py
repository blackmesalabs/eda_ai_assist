#!/usr/bin/env python3
"""
ashchat.py - Tkinter-based GUI frontend for eda_ai_assist.py API
Provides a ChatBot-like interface with file management and session control.
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, simpledialog
import sys
import os
from typing import List, Dict
import threading
import queue

# Import the eda_ai_assist API
try:
    from eda_ai_assist import api_eda_ai_assist
except ImportError:
    print("Error: Could not import eda_ai_assist module. Ensure eda_ai_assist.py is in the same directory.")
    sys.exit(1)


class AshChat:
    def __init__(self, root):
        self.root = root
        self.root.geometry("900x700")
        self.root.minsize(600, 400)

        # Determine default font size based on OS
        if sys.platform.startswith("win32"):
            self.default_font_size = 10
            self.default_gui_font_size = 10
        else:
            self.default_font_size = 11
            self.default_gui_font_size = 11
        self.current_font_size = self.default_font_size
        self.current_gui_font_size = self.default_gui_font_size

        # Load model list from site_model_list.txt
        self.model_config = self._load_model_config()
        self.current_model_nickname = None

        # AI session
        self.ai = api_eda_ai_assist()
        self.ai.open_ai_session()
        if not self.ai.provider:
            messagebox.showerror("Error", "Failed to initialize AI session.")
            sys.exit(1)

        # Detect current model
        self.current_model_nickname = self._get_current_model_nickname()

        # Set window title with provider and model
        provider = self.ai.cfg.get("ASH_PROVIDER", "unknown")
        model = self.ai.cfg.get("ASH_MODEL", "unknown")
        self.root.title(f"AshChat - {provider}:{model}")

        # Queue for thread-safe message passing
        self.response_queue = queue.Queue()

        # Chat history (display only, cleared on restart)
        self.chat_history: List[str] = []

        # Loaded files list (automatically included in prompts)
        self.loaded_files: List[str] = []

        # Track input frame for dynamic resizing
        self.input_frame = None
        self.input_text_frame = None
        self.input_multiline_expanded = False

        # UI component references for font updates
        self.ui_components = {
            "labels": [],
            "buttons": [],
            "menus": []
        }

        # Menu variable for tracking selected model
        self.selected_model_var = tk.StringVar()

        # Setup GUI
        self._setup_menu()
        self._setup_loaded_files_panel()
        self._setup_chat_display()
        self._setup_input_area()
        self._setup_status_bar()

        # Process queued responses
        self._process_queue()

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
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load File", command=self._load_file)
        file_menu.add_command(label="Unload File", command=self._unload_file)
        file_menu.add_separator()
        file_menu.add_command(label="Save Chat Transcript", command=self._save_transcript)
        file_menu.add_command(label="Save Selected Text", command=self._save_selected)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy", command=self._copy_text)
        edit_menu.add_command(label="Paste", command=self._paste_text)
        edit_menu.add_command(label="Select All", command=self._select_all)
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Chat", command=self._clear_chat)

        # Format menu
        format_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Format", menu=format_menu)
        format_menu.add_command(label="Font Size 8pt", command=lambda: self._set_font_size(8))
        format_menu.add_command(label="Font Size 9pt", command=lambda: self._set_font_size(9))
        format_menu.add_command(label="Font Size 10pt", command=lambda: self._set_font_size(10))
        format_menu.add_command(label="Font Size 11pt", command=lambda: self._set_font_size(11))
        format_menu.add_command(label="Font Size 12pt", command=lambda: self._set_font_size(12))
        format_menu.add_command(label="Font Size 13pt", command=lambda: self._set_font_size(13))
        format_menu.add_command(label="Font Size 14pt", command=lambda: self._set_font_size(14))
        format_menu.add_command(label="Font Size 15pt", command=lambda: self._set_font_size(15))
        format_menu.add_command(label="Font Size 16pt", command=lambda: self._set_font_size(16))
        format_menu.add_command(label="Font Size 17pt", command=lambda: self._set_font_size(17))
        format_menu.add_command(label="Font Size 18pt", command=lambda: self._set_font_size(18))
        format_menu.add_separator()
        format_menu.add_command(label="Reset to Default", command=self._reset_font_size)

        # AI menu (model selection)
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)

        if self.model_config:
            # Build two-tier menu: provider -> model nickname
            providers = {}
            for nickname, config in self.model_config.items():
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
                        command=lambda n=nickname: self._switch_model(n)
                    )

            # Set current selection
            if self.current_model_nickname:
                self.selected_model_var.set(self.current_model_nickname)
        else:
            ai_menu.add_command(label="(No models configured)", state=tk.DISABLED)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="View Help", command=self._show_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

    def _switch_model(self, nickname: str):
        """Switch to a different AI model"""
        if nickname not in self.model_config:
            messagebox.showerror("Error", f"Model {nickname} not found in configuration.")
            return

        config = self.model_config[nickname]

        if messagebox.askyesno("Confirm", f"Switch to model {nickname}?"):
            # Close current session
            if self.ai.provider:
                self.ai.close_ai_session()

            # Apply new configuration
            for key, value in config.items():
                os.environ[key] = value
                self.ai.cfg[key] = value

            # Open new session with new config
            self.ai.open_ai_session()
            if not self.ai.provider:
                messagebox.showerror("Error", "Failed to initialize new AI session.")
                return

            # Update current model
            self.current_model_nickname = nickname

            # Update window title
            provider = self.ai.cfg.get("ASH_PROVIDER", "unknown")
            model = self.ai.cfg.get("ASH_MODEL", "unknown")
            self.root.title(f"AshChat - {provider}:{model}")

            # Append system message
            self._append_chat("system", f"Switched to model: {nickname}")
            self._update_status()
        else:
            # Revert selection
            if self.current_model_nickname:
                self.selected_model_var.set(self.current_model_nickname)

    def _setup_loaded_files_panel(self):
        # Frame for loaded files
        files_frame = tk.LabelFrame(
            self.root,
            text="Loaded Files (auto-included in prompts)",
            font=("Courier", self.current_gui_font_size, "bold")
        )
        files_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.ui_components["labels"].append(("files_frame_label", files_frame))

        # Scrollable listbox for loaded files
        files_scroll = tk.Scrollbar(files_frame)
        files_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.loaded_files_listbox = tk.Listbox(
            files_frame,
            height=3,
            font=("Courier", self.current_font_size),
            yscrollcommand=files_scroll.set
        )
        self.loaded_files_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        files_scroll.config(command=self.loaded_files_listbox.yview)

    def _setup_chat_display(self):
        # Frame for chat display
        chat_frame = tk.Frame(self.root)
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Label
        chat_label = tk.Label(
            chat_frame,
            text="Chat History",
            font=("Courier", self.current_gui_font_size, "bold")
        )
        chat_label.pack(anchor=tk.W)
        self.ui_components["labels"].append(("chat_label", chat_label))

        # Chat display area (read-only)
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            height=20,
            width=100,
#           state=tk.DISABLED,
            state=tk.NORMAL,
            font=("Courier", self.current_font_size),
            bg="#f0f0f0",
            wrap=tk.WORD
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Bind keys to prevent editing but allow selection and copying
        self.chat_display.bind("<Key>", lambda e: "break")
        self.chat_display.bind("<Control-c>", lambda e: self._copy_text())
        self.chat_display.bind("<Control-v>", lambda e: "break")
        self.chat_display.bind("<Delete>", lambda e: "break")
        self.chat_display.bind("<BackSpace>", lambda e: "break")

#       self.chat_display.bind("<Key>", lambda e: "break" if e.state == 0 else None)
#       self.chat_display.bind("<Control-c>", self._handle_copy)
#       self.chat_display.bind("<Control-v>", lambda e: "break")
#       self.chat_display.bind("<Delete>", lambda e: "break")
#       self.chat_display.bind("<BackSpace>", lambda e: "break")


    def _handle_copy(self, event=None):
        try:
            selected = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected:
                self.chat_display.clipboard_clear()
                self.chat_display.clipboard_append(selected)
        except tk.TclError:
            pass
        return "break"


    def _setup_input_area(self):
        # Frame for input and buttons
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=5, pady=5)

        # Label
        input_label = tk.Label(
            self.input_frame,
            text="Prompt (Ctrl+Enter to send, Ctrl+D for multi-line)",
            font=("Courier", self.current_gui_font_size)
        )
        input_label.pack(anchor=tk.W)
        self.ui_components["labels"].append(("input_label", input_label))

        # Frame for input text with scrollbars
        self.input_text_frame = tk.Frame(self.input_frame)
        self.input_text_frame.pack(fill=tk.BOTH, expand=True)

        # Vertical scrollbar
        v_scroll = tk.Scrollbar(self.input_text_frame)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        h_scroll = tk.Scrollbar(self.input_text_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Input area (multi-line) with scrollbars
        self.input_text = tk.Text(
            self.input_text_frame,
            height=5,
            width=100,
            font=("Courier", self.current_font_size),
            wrap=tk.NONE,
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        v_scroll.config(command=self.input_text.yview)
        h_scroll.config(command=self.input_text.xview)

        self.input_text.bind("<Control-Return>", lambda e: self._send_message())
        self.input_text.bind("<Control-d>", lambda e: self._toggle_multiline())

        # Button frame
        button_frame = tk.Frame(self.input_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.send_button = tk.Button(
            button_frame,
            text="Send",
            command=self._send_message,
            bg="#4CAF50",
            fg="white",
            font=("Courier", self.current_gui_font_size, "bold"),
            padx=20,
            pady=5
        )
        self.send_button.pack(side=tk.RIGHT, padx=5)
        self.ui_components["buttons"].append(("send_button", self.send_button))

        self.flush_button = tk.Button(
            button_frame,
            text="Flush Session",
            command=self._flush_session,
            bg="#FF9800",
            fg="white",
            font=("Courier", self.current_gui_font_size, "bold"),
            padx=20,
            pady=5
        )
        self.flush_button.pack(side=tk.RIGHT, padx=5)
        self.ui_components["buttons"].append(("flush_button", self.flush_button))

    def _setup_status_bar(self):
        self.status_bar = tk.Label(
            self.root,
            text=self._get_initial_status(),
            font=("Courier", self.current_gui_font_size),
            bg="#e0e0e0",
            anchor=tk.W,
            padx=5,
            pady=3
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.ui_components["labels"].append(("status_bar", self.status_bar))

    def _get_initial_status(self) -> str:
        return f"Tokens: 0 | Files: 0 | Response Time: N/A"

    def _update_status(self):
        if self.ai.provider:
            tokens = self.ai.token_cnt_total
            files = len(self.ai.session_file_list)
            response_time = self.ai.last_response_time
            status_text = f"Tokens: {tokens:,} | Files: {files} | Response Time: {response_time:.2f}s"
            self.status_bar.config(text=status_text)
        else:
            status_text = f"Status: AI session not active"
            self.status_bar.config(text=status_text)

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

    def _update_loaded_files_display(self):
        self.loaded_files_listbox.delete(0, tk.END)
        for file_path in self.loaded_files:
            basename = os.path.basename(file_path)
            self.loaded_files_listbox.insert(tk.END, basename)

    def _send_message(self):
        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty Prompt", "Please enter a message.")
            return

        if not self.ai.provider:
            messagebox.showerror("Error", "AI session is not active.")
            return

        # If files are loaded, prepend them to the prompt
        if self.loaded_files:
            file_list_str = ", ".join(os.path.basename(f) for f in self.loaded_files)
            enhanced_prompt = f"Here are loaded files: {file_list_str}.\n\n{prompt}"
        else:
            enhanced_prompt = prompt

        self._append_chat("user", prompt)
        self.input_text.delete("1.0", tk.END)
#       self.send_button.config(state=tk.DISABLED)
#       self.input_text.config(state=tk.DISABLED)

        # Run AI request in background thread
        thread = threading.Thread(target=self._ai_request_thread, args=(enhanced_prompt,), daemon=True)
        thread.start()

    def _ai_request_thread(self, prompt: str):
        try:
            response = self.ai.ask_ai(prompt)
            warnings = self.ai.get_warnings()
            self.response_queue.put(("response", response, warnings))
        except Exception as e:
            self.response_queue.put(("error", str(e), None))

    def _process_queue(self):
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
        finally:
            self.send_button.config(state=tk.NORMAL)
            self.input_text.config(state=tk.NORMAL)
            self.input_text.focus()
            self.root.after(100, self._process_queue)

    def _toggle_multiline(self):
        self.input_frame.update_idletasks()
        frame_height = self.input_frame.winfo_height()

        if not self.input_multiline_expanded:
            self.input_text.config(height=15)
            self.input_multiline_expanded = True
        else:
            self.input_text.config(height=5)
            self.input_multiline_expanded = False

    def _set_font_size(self, size: int):
        if 8 <= size <= 18:
            self.current_font_size = size
            self.current_gui_font_size = size

            # Update chat display and input text
            self.chat_display.config(font=("Courier", size))
            self.input_text.config(font=("Courier", size))
            self.loaded_files_listbox.config(font=("Courier", size))

            # Update all GUI component fonts
            for component_type, components in self.ui_components.items():
                if component_type == "labels":
                    for label_name, label_widget in components:
                        if label_name == "files_frame_label":
                            label_widget.config(font=("Courier", size, "bold"))
                        else:
                            label_widget.config(font=("Courier", size, "bold"))
                elif component_type == "buttons":
                    for button_name, button_widget in components:
                        button_widget.config(font=("Courier", size, "bold"))

    def _reset_font_size(self):
        self.current_font_size = self.default_font_size
        self.current_gui_font_size = self.default_gui_font_size
        self._set_font_size(self.default_font_size)

    def _load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a file to load",
            filetypes=[("All files", "*.*"), ("Text files", "*.txt"), ("Verilog", "*.v"), ("VCD", "*.vcd")]
        )
        if file_path:
            if file_path not in self.loaded_files:
                self.loaded_files.append(file_path)
                if file_path not in self.ai.session_file_list:
                    self.ai.session_file_list.append(file_path)
                self._update_loaded_files_display()
                self._append_chat("system", f"Loaded file: {os.path.basename(file_path)}")
                self._update_status()
            else:
                messagebox.showinfo("Info", f"{os.path.basename(file_path)} is already loaded.")

    def _unload_file(self):
        if not self.loaded_files:
            messagebox.showinfo("Info", "No files to unload.")
            return

        unload_window = tk.Toplevel(self.root)
        unload_window.title("Unload File")
        unload_window.geometry("400x200")

        title_label = tk.Label(
            unload_window,
            text="Select file(s) to unload:",
            font=("Courier", self.current_gui_font_size)
        )
        title_label.pack(pady=10)

        files_to_remove = []

        def toggle_file(file_path):
            if file_path in files_to_remove:
                files_to_remove.remove(file_path)
            else:
                files_to_remove.append(file_path)

        for file_path in self.loaded_files:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(
                unload_window,
                text=os.path.basename(file_path),
                command=lambda fp=file_path: toggle_file(fp),
                font=("Courier", self.current_gui_font_size)
            )
            chk.pack(anchor=tk.W, padx=20)

        def confirm_unload():
            if files_to_remove:
                for file_path in files_to_remove:
                    self.loaded_files.remove(file_path)
                    if file_path in self.ai.session_file_list:
                        self.ai.session_file_list.remove(file_path)
                self._update_loaded_files_display()
                self._append_chat("system", f"Unloaded {len(files_to_remove)} file(s).")
                self._update_status()
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

    def _save_transcript(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                content = self.chat_display.get("1.0", tk.END)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Chat transcript saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save transcript: {e}")

    def _save_selected(self):
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

    def _copy_text(self):
        try:
            selected = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected:
                self.root.clipboard_clear()
                self.root.clipboard_append(selected)
#               self.root.clipboard_update()
            else:
                messagebox.showwarning("No Selection", "Please select text to copy.")
        except tk.TclError:
            messagebox.showwarning("No Selection", "Please select text to copy.")

#       try:
#           self.chat_display.config(state=tk.NORMAL)
#           try:
#               sel_start = self.chat_display.index(tk.SEL_FIRST)
#               sel_end = self.chat_display.index(tk.SEL_LAST)
#               selected = self.chat_display.get(sel_start, sel_end)
#           finally:
#               self.chat_display.config(state=tk.DISABLED)
#           
#           if selected:
#               self.root.clipboard_clear()
#               self.root.clipboard_append(selected)
#               self.root.clipboard_update()
#           else:
#               messagebox.showwarning("No Selection", "Please select text to copy.")
#       except tk.TclError:
#           self.chat_display.config(state=tk.DISABLED)
#           messagebox.showwarning("No Selection", "Please select text to copy.")
#
#       try:
#           selected = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
#           self.root.clipboard_clear()
#           self.root.clipboard_append(selected)
#       except tk.TclError:
#           messagebox.showwarning("No Selection", "Please select text to copy.")

#       try:
#           self.chat_display.config(state=tk.NORMAL)
#           selected = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
#           self.chat_display.config(state=tk.DISABLED)
#           self.root.clipboard_clear()
#           self.root.clipboard_append(selected)
#           self.root.clipboard_update()
#       except tk.TclError:
#           self.chat_display.config(state=tk.DISABLED)
#           messagebox.showwarning("No Selection", "Please select text to copy.")


    def _paste_text(self):
        try:
            text = self.root.clipboard_get()
            self.input_text.insert(tk.INSERT, text)
        except tk.TclError:
            messagebox.showerror("Error", "Failed to paste from clipboard.")

    def _select_all(self):
        self.chat_display.tag_add(tk.SEL, "1.0", tk.END)
        self.chat_display.mark_set(tk.INSERT, "1.0")
        self.chat_display.see(tk.INSERT)

    def _clear_chat(self):
        if messagebox.askyesno("Confirm", "Clear all chat history?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
#           self.chat_display.config(state=tk.DISABLED)
            self.chat_history.clear()

    def _flush_session(self):
        if messagebox.askyesno("Confirm", "Flush AI session, clear chat, and unload all files?"):
            # Close AI session
            self.ai.close_ai_session()

            # Clear chat display
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
#           self.chat_display.config(state=tk.DISABLED)
            self.chat_history.clear()

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
                messagebox.showerror("Error", "Failed to restart AI session.")

    def _show_help(self):
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


    def _show_about(self):
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
- Amazon Bedrock (Claude)
"""
        self._show_scrollable_dialog("About AshChat", about_text, width=70, height=18)

    def _show_scrollable_dialog(self, title: str, text: str, width: int = 80, height: int = 30):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry(f"{width*8}x{height*14}")

        # Scrollable text area
        text_area = scrolledtext.ScrolledText(
            dialog,
            width=width,
            height=height,
            font=("Courier", self.current_gui_font_size),
            wrap=tk.WORD,
            state=tk.NORMAL
        )
        text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_area.insert("1.0", text)
        text_area.config(state=tk.DISABLED)

        close_button = tk.Button(
            dialog,
            text="Close",
            command=dialog.destroy,
            font=("Courier", self.current_gui_font_size),
            padx=20,
            pady=5
        )
        close_button.pack(pady=5)

    def _on_exit(self):
#       if messagebox.askyesno("Exit", "Close AshChat and terminate AI session?"):
#           if self.ai.provider:
#               self.ai.close_ai_session()
#           self.root.quit()

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

        if messagebox.askyesno("Exit", msg):
            if self.ai.provider:
                self.ai.close_ai_session()
            self.root.quit()



def main():
    root = tk.Tk()
    app = AshChat(root)
    root.mainloop()


if __name__ == "__main__":
    main()
