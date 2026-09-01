"""
Ollama Chat - prosty agent z interfejsem graficznym (tkinter).

Łączy się z serwerem Ollama, pozwala rozmawiać z modelem i przechowuje
historię wielu czatów (zapisywaną na dysku w pliku JSON).

Wymaga tylko biblioteki standardowej Pythona (tkinter + urllib).
"""

import json
import queue
import threading
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

# --- Konfiguracja ---------------------------------------------------------

OLLAMA_URL = "http://192.168.100.52:11434"
DEFAULT_MODEL = "llama3:latest"
HISTORY_FILE = Path(__file__).with_name("chat_history.json")
REQUEST_TIMEOUT = 300  # sekundy


# --- Warstwa komunikacji z Ollamą ----------------------------------------

class OllamaClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def list_models(self):
        """Zwraca listę nazw modeli dostępnych na serwerze."""
        url = f"{self.base_url}/api/tags"
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]

    def chat_stream(self, model, messages):
        """
        Wysyła zapytanie do /api/chat w trybie strumieniowym.
        Generator zwraca kolejne fragmenty tekstu odpowiedzi.
        """
        url = f"{self.base_url}/api/chat"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "message" in obj and "content" in obj["message"]:
                    yield obj["message"]["content"]
                if obj.get("done"):
                    break


# --- Przechowywanie historii ---------------------------------------------

class HistoryStore:
    def __init__(self, path):
        self.path = path
        self.chats = []  # lista: {"id", "title", "created", "messages": [...]}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self.chats = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.chats = []

    def save(self):
        try:
            self.path.write_text(
                json.dumps(self.chats, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            print("Nie udało się zapisać historii:", e)

    def new_chat(self):
        chat = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "title": "Nowy czat",
            "created": datetime.now().isoformat(timespec="seconds"),
            "messages": [],
        }
        self.chats.insert(0, chat)
        return chat

    def delete(self, chat_id):
        self.chats = [c for c in self.chats if c["id"] != chat_id]
        self.save()


# --- Interfejs graficzny --------------------------------------------------

class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ollama Chat")
        self.geometry("980x640")
        self.minsize(760, 480)

        self.client = OllamaClient(OLLAMA_URL)
        self.store = HistoryStore(HISTORY_FILE)
        self.current_chat = None
        self.streaming = False
        self.msg_queue = queue.Queue()

        self._build_ui()
        self._refresh_chat_list()
        self._load_models_async()

        # Jeśli nie ma żadnego czatu, utwórz pierwszy.
        if not self.store.chats:
            self._new_chat()
        else:
            self._select_chat(self.store.chats[0]["id"])

        self.after(50, self._process_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- Budowa layoutu --------------------------------------------------

    def _build_ui(self):
        # Górny pasek: model + status.
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(
            top, textvariable=self.model_var, width=28, state="readonly"
        )
        self.model_combo.pack(side=tk.LEFT, padx=(4, 12))

        self.status_var = tk.StringVar(value=f"Serwer: {OLLAMA_URL}")
        ttk.Label(top, textvariable=self.status_var, foreground="#666").pack(
            side=tk.LEFT
        )

        ttk.Button(top, text="Odśwież modele", command=self._load_models_async).pack(
            side=tk.RIGHT
        )

        # Główny obszar: panel z listą czatów + panel rozmowy.
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Lewa kolumna - lista czatów.
        left = ttk.Frame(main, width=220)
        main.add(left, weight=0)

        btns = ttk.Frame(left)
        btns.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Button(btns, text="+ Nowy czat", command=self._new_chat).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(btns, text="Usuń", command=self._delete_chat, width=6).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        self.chat_listbox = tk.Listbox(left, activestyle="none", exportselection=False)
        self.chat_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.chat_listbox.bind("<<ListboxSelect>>", self._on_chat_selected)

        # Prawa kolumna - rozmowa.
        right = ttk.Frame(main)
        main.add(right, weight=1)

        self.text = tk.Text(
            right, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10),
            padx=10, pady=8, background="#fafafa",
        )
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        vscroll = ttk.Scrollbar(self.text, command=self.text.yview)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=vscroll.set)

        self.text.tag_configure(
            "user", foreground="#0b5cad", font=("Segoe UI", 10, "bold"),
            spacing1=8,
        )
        self.text.tag_configure(
            "assistant", foreground="#1a7f37", font=("Segoe UI", 10, "bold"),
            spacing1=8,
        )
        self.text.tag_configure("body", foreground="#222", spacing3=4, lmargin1=4, lmargin2=4)
        self.text.tag_configure("error", foreground="#b00020")

        # Dolny pasek - pole wprowadzania.
        bottom = ttk.Frame(right)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

        self.entry = tk.Text(bottom, height=3, wrap=tk.WORD, font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<Shift-Return>", lambda e: None)  # nowa linia

        self.send_btn = ttk.Button(bottom, text="Wyślij", command=self._send)
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))

    # ---- Obsługa modeli --------------------------------------------------

    def _load_models_async(self):
        def worker():
            try:
                models = self.client.list_models()
                self.msg_queue.put(("models", models))
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.msg_queue.put(("models_error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    # ---- Lista czatów ----------------------------------------------------

    def _refresh_chat_list(self):
        self.chat_listbox.delete(0, tk.END)
        for chat in self.store.chats:
            self.chat_listbox.insert(tk.END, f"  {chat['title']}")

    def _new_chat(self):
        chat = self.store.new_chat()
        self.store.save()
        self._refresh_chat_list()
        self._select_chat(chat["id"])
        self.entry.focus_set()

    def _delete_chat(self):
        if not self.current_chat:
            return
        if not messagebox.askyesno("Usuń czat", "Na pewno usunąć ten czat?"):
            return
        self.store.delete(self.current_chat["id"])
        self.current_chat = None
        self._refresh_chat_list()
        if self.store.chats:
            self._select_chat(self.store.chats[0]["id"])
        else:
            self._new_chat()

    def _on_chat_selected(self, _event):
        sel = self.chat_listbox.curselection()
        if not sel:
            return
        chat = self.store.chats[sel[0]]
        if not self.current_chat or chat["id"] != self.current_chat["id"]:
            self._select_chat(chat["id"])

    def _select_chat(self, chat_id):
        chat = next((c for c in self.store.chats if c["id"] == chat_id), None)
        if not chat:
            return
        self.current_chat = chat
        # Zaznacz w liście.
        idx = self.store.chats.index(chat)
        self.chat_listbox.selection_clear(0, tk.END)
        self.chat_listbox.selection_set(idx)
        self._render_conversation()

    # ---- Renderowanie rozmowy -------------------------------------------

    def _render_conversation(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        for msg in self.current_chat["messages"]:
            self._append_message(msg["role"], msg["content"], save=False)
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)

    def _append_message(self, role, content, save=True):
        self.text.configure(state=tk.NORMAL)
        label = "Ty" if role == "user" else "Model"
        tag = "user" if role == "user" else "assistant"
        self.text.insert(tk.END, f"{label}\n", tag)
        self.text.insert(tk.END, content + "\n", "body")
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)

    # ---- Wysyłanie wiadomości -------------------------------------------

    def _on_enter(self, event):
        # Enter wysyła, Shift+Enter dodaje nową linię.
        if event.state & 0x0001:  # Shift wciśnięty
            return
        self._send()
        return "break"

    def _send(self):
        if self.streaming:
            return
        text = self.entry.get("1.0", tk.END).strip()
        if not text:
            return
        if not self.current_chat:
            self._new_chat()

        self.entry.delete("1.0", tk.END)

        # Dodaj wiadomość użytkownika.
        self.current_chat["messages"].append({"role": "user", "content": text})
        self._append_message("user", text)

        # Ustaw tytuł czatu z pierwszej wiadomości.
        if self.current_chat["title"] == "Nowy czat":
            self.current_chat["title"] = (text[:30] + "…") if len(text) > 30 else text
            self._refresh_chat_list()
            self._select_chat(self.current_chat["id"])

        self.store.save()

        # Zacznij nagłówek odpowiedzi modelu.
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, "Model\n", "assistant")
        self._assistant_mark = self.text.index(tk.END)
        self.text.configure(state=tk.DISABLED)

        self.streaming = True
        self.send_btn.configure(state=tk.DISABLED)
        self.status_var.set("Model odpowiada…")

        model = self.model_var.get() or DEFAULT_MODEL
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in self.current_chat["messages"]
        ]

        threading.Thread(
            target=self._stream_worker, args=(model, messages), daemon=True
        ).start()

    def _stream_worker(self, model, messages):
        try:
            for chunk in self.client.chat_stream(model, messages):
                self.msg_queue.put(("chunk", chunk))
            self.msg_queue.put(("done", None))
        except urllib.error.URLError as e:
            self.msg_queue.put(("stream_error", f"Błąd połączenia: {e.reason}"))
        except Exception as e:  # noqa: BLE001 - pokazujemy każdy błąd w GUI
            self.msg_queue.put(("stream_error", str(e)))

    # ---- Pętla przetwarzania zdarzeń z wątków ---------------------------

    def _process_queue(self):
        try:
            while True:
                kind, data = self.msg_queue.get_nowait()
                if kind == "models":
                    self._on_models_loaded(data)
                elif kind == "models_error":
                    self.status_var.set(f"Nie można pobrać modeli: {data}")
                elif kind == "chunk":
                    self._on_chunk(data)
                elif kind == "done":
                    self._on_stream_done()
                elif kind == "stream_error":
                    self._on_stream_error(data)
        except queue.Empty:
            pass
        self.after(50, self._process_queue)

    def _on_models_loaded(self, models):
        if models:
            self.model_combo.configure(values=models)
            if self.model_var.get() not in models:
                self.model_var.set(models[0])
            self.status_var.set(f"Połączono • {len(models)} modeli")
        else:
            self.status_var.set("Serwer nie zwrócił żadnych modeli")

    def _on_chunk(self, chunk):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, chunk, "body")
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)
        # Bufor odpowiedzi.
        self._assistant_buffer = getattr(self, "_assistant_buffer", "") + chunk

    def _on_stream_done(self):
        content = getattr(self, "_assistant_buffer", "")
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, "\n", "body")
        self.text.configure(state=tk.DISABLED)
        self.current_chat["messages"].append(
            {"role": "assistant", "content": content}
        )
        self.store.save()
        self._assistant_buffer = ""
        self.streaming = False
        self.send_btn.configure(state=tk.NORMAL)
        self.status_var.set("Gotowe")
        self.entry.focus_set()

    def _on_stream_error(self, message):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, f"\n[{message}]\n", "error")
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)
        self._assistant_buffer = ""
        self.streaming = False
        self.send_btn.configure(state=tk.NORMAL)
        self.status_var.set("Błąd")

    def _on_close(self):
        self.store.save()
        self.destroy()


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
