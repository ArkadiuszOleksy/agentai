"""
AnythingLLM Chat - agent z interfejsem graficznym (tkinter).

Łączy się z serwerem AnythingLLM, pozwala rozmawiać z wybranym workspace,
DOŁĄCZAĆ dokumenty (zdjęcia / skany / PDF) i prosić model o przepisanie
1:1 treści - również pisma ręcznego.

Historia wielu czatów zapisywana jest lokalnie w pliku JSON.
Wymaga tylko biblioteki standardowej Pythona (tkinter + urllib).

WAŻNE: przepisywanie treści ze zdjęć/skanów działa tylko wtedy, gdy
workspace w AnythingLLM korzysta z modelu obsługującego obraz (vision),
np. llava, llama3.2-vision, gpt-4o itp.
"""

import base64
import json
import mimetypes
import queue
import threading
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# --- Konfiguracja domyślna -----------------------------------------------

DEFAULT_BASE_URL = "http://192.168.100.19:3001"
DEFAULT_API_KEY = "0JS9SX6-R3944KD-P9F192N-KGREWMR"

CONFIG_FILE = Path(__file__).with_name("anythingllm_config.json")
HISTORY_FILE = Path(__file__).with_name("anythingllm_history.json")
REQUEST_TIMEOUT = 600  # sekundy (przepisywanie dokumentów bywa wolne)

# Domyślne polecenie wstawiane do pola, gdy dołączysz dokument.
TRANSCRIBE_PROMPT = (
    "Przepisz dokładnie 1:1 całą treść z załączonego dokumentu. "
    "Odwzoruj zarówno tekst drukowany, jak i pismo ręczne. "
    "Nie streszczaj, nie poprawiaj, nie tłumacz - zachowaj oryginalny "
    "układ, interpunkcję i podział na akapity. Jeśli czegoś nie da się "
    "odczytać, oznacz to jako [nieczytelne]."
)


# --- Klient AnythingLLM ---------------------------------------------------

class AnythingLLMClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self, extra=None):
        h = {"Authorization": f"Bearer {self.api_key}"}
        if extra:
            h.update(extra)
        return h

    def list_workspaces(self):
        """Zwraca listę (name, slug) dostępnych workspace."""
        url = f"{self.base_url}/api/v1/workspaces"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [(w["name"], w["slug"]) for w in data.get("workspaces", [])]

    def stream_chat(self, slug, message, attachments=None):
        """
        Strumieniowa rozmowa z workspace.
        Generator zwraca kolejne fragmenty odpowiedzi (tekst).
        attachments: lista {"name","mime","contentString"} (data-URI base64).
        """
        url = f"{self.base_url}/api/v1/workspace/{slug}/stream-chat"
        body = {
            "message": message,
            "mode": "chat",
            "attachments": attachments or [],
        }
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers=self._headers({"Content-Type": "application/json"}),
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if not chunk:
                    continue
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    raise RuntimeError(str(obj["error"]))
                text = obj.get("textResponse") or ""
                if text:
                    yield text
                if obj.get("close"):
                    break


# --- Ustawienia i historia -----------------------------------------------

class Config:
    def __init__(self, path):
        self.path = path
        self.base_url = DEFAULT_BASE_URL
        self.api_key = DEFAULT_API_KEY
        self.workspace_slug = ""
        self.load()

    def load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.base_url = d.get("base_url", self.base_url)
                self.api_key = d.get("api_key", self.api_key)
                self.workspace_slug = d.get("workspace_slug", "")
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        try:
            self.path.write_text(
                json.dumps(
                    {
                        "base_url": self.base_url,
                        "api_key": self.api_key,
                        "workspace_slug": self.workspace_slug,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as e:
            print("Nie udało się zapisać konfiguracji:", e)


class HistoryStore:
    def __init__(self, path):
        self.path = path
        self.chats = []
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
        self.title("AnythingLLM Chat")
        self.geometry("1040x680")
        self.minsize(820, 520)

        self.config_store = Config(CONFIG_FILE)
        self.client = AnythingLLMClient(
            self.config_store.base_url, self.config_store.api_key
        )
        self.store = HistoryStore(HISTORY_FILE)
        self.current_chat = None
        self.streaming = False
        self.msg_queue = queue.Queue()
        self.workspaces = []           # [(name, slug)]
        self.pending_attachments = []  # [{"name","mime","contentString"}]
        self._assistant_buffer = ""

        self._build_ui()
        self._refresh_chat_list()
        self._test_connection_async()

        if not self.store.chats:
            self._new_chat()
        else:
            self._select_chat(self.store.chats[0]["id"])

        self.after(50, self._process_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- Layout ----------------------------------------------------------

    def _build_ui(self):
        # Pasek połączenia.
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Adres:").pack(side=tk.LEFT)
        self.url_var = tk.StringVar(value=self.config_store.base_url)
        ttk.Entry(top, textvariable=self.url_var, width=26).pack(
            side=tk.LEFT, padx=(4, 10)
        )

        ttk.Label(top, text="Klucz API:").pack(side=tk.LEFT)
        self.key_var = tk.StringVar(value=self.config_store.api_key)
        ttk.Entry(top, textvariable=self.key_var, width=26, show="•").pack(
            side=tk.LEFT, padx=(4, 10)
        )

        ttk.Label(top, text="Workspace:").pack(side=tk.LEFT)
        self.ws_var = tk.StringVar()
        self.ws_combo = ttk.Combobox(
            top, textvariable=self.ws_var, width=20, state="readonly"
        )
        self.ws_combo.pack(side=tk.LEFT, padx=(4, 10))
        self.ws_combo.bind("<<ComboboxSelected>>", self._on_ws_selected)

        ttk.Button(top, text="Połącz / Odśwież", command=self._test_connection_async).pack(
            side=tk.LEFT
        )

        # Pasek statusu.
        status_bar = ttk.Frame(self, padding=(8, 0))
        status_bar.pack(side=tk.TOP, fill=tk.X)
        self.status_var = tk.StringVar(value="Łączenie…")
        ttk.Label(status_bar, textvariable=self.status_var, foreground="#666").pack(
            side=tk.LEFT
        )

        # Główny obszar.
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

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

        self.text.tag_configure("user", foreground="#0b5cad", font=("Segoe UI", 10, "bold"), spacing1=8)
        self.text.tag_configure("assistant", foreground="#1a7f37", font=("Segoe UI", 10, "bold"), spacing1=8)
        self.text.tag_configure("body", foreground="#222", spacing3=4, lmargin1=4, lmargin2=4)
        self.text.tag_configure("attach", foreground="#8a5a00")
        self.text.tag_configure("error", foreground="#b00020")

        # Pasek załączników.
        attach_bar = ttk.Frame(right)
        attach_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 2))
        ttk.Button(attach_bar, text="📎 Dołącz dokument", command=self._attach_files).pack(side=tk.LEFT)
        self.attach_label = ttk.Label(attach_bar, text="Brak załączników", foreground="#888")
        self.attach_label.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(attach_bar, text="Wyczyść załączniki", command=self._clear_attachments).pack(side=tk.RIGHT)

        # Pole wprowadzania.
        bottom = ttk.Frame(right)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.entry = tk.Text(bottom, height=3, wrap=tk.WORD, font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.send_btn = ttk.Button(bottom, text="Wyślij", command=self._send)
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))

    # ---- Połączenie / workspace -----------------------------------------

    def _apply_connection_fields(self):
        self.config_store.base_url = self.url_var.get().strip()
        self.config_store.api_key = self.key_var.get().strip()
        self.client = AnythingLLMClient(
            self.config_store.base_url, self.config_store.api_key
        )

    def _test_connection_async(self):
        self._apply_connection_fields()
        self.status_var.set("Łączenie…")

        def worker():
            try:
                ws = self.client.list_workspaces()
                self.msg_queue.put(("workspaces", ws))
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    self.msg_queue.put(("conn_error",
                        "Nieprawidłowy klucz API (403). Wygeneruj nowy w AnythingLLM: "
                        "Settings → Tools → Developer API → Generate New API Key."))
                else:
                    self.msg_queue.put(("conn_error", f"HTTP {e.code}: {e.reason}"))
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.msg_queue.put(("conn_error", f"Brak połączenia z serwerem: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ws_selected(self, _event=None):
        name = self.ws_var.get()
        slug = next((s for n, s in self.workspaces if n == name), "")
        self.config_store.workspace_slug = slug
        self.config_store.save()

    def _current_slug(self):
        name = self.ws_var.get()
        return next((s for n, s in self.workspaces if n == name), "")

    # ---- Załączniki ------------------------------------------------------

    def _attach_files(self):
        paths = filedialog.askopenfilenames(
            title="Wybierz dokumenty do przepisania",
            filetypes=[
                ("Obrazy i dokumenty", "*.png *.jpg *.jpeg *.webp *.gif *.bmp *.pdf *.txt"),
                ("Obrazy", "*.png *.jpg *.jpeg *.webp *.gif *.bmp"),
                ("PDF", "*.pdf"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not paths:
            return
        for p in paths:
            path = Path(p)
            try:
                raw = path.read_bytes()
            except OSError as e:
                messagebox.showerror("Błąd", f"Nie można wczytać {path.name}: {e}")
                continue
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            b64 = base64.b64encode(raw).decode("ascii")
            self.pending_attachments.append({
                "name": path.name,
                "mime": mime,
                "contentString": f"data:{mime};base64,{b64}",
            })
        self._update_attach_label()

        # Podpowiedz polecenie transkrypcji, jeśli pole jest puste.
        if self.pending_attachments and not self.entry.get("1.0", tk.END).strip():
            self.entry.insert("1.0", TRANSCRIBE_PROMPT)

    def _clear_attachments(self):
        self.pending_attachments = []
        self._update_attach_label()

    def _update_attach_label(self):
        n = len(self.pending_attachments)
        if n == 0:
            self.attach_label.configure(text="Brak załączników", foreground="#888")
        else:
            names = ", ".join(a["name"] for a in self.pending_attachments)
            self.attach_label.configure(
                text=f"{n} plik(ów): {names}", foreground="#8a5a00"
            )

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
        idx = self.store.chats.index(chat)
        self.chat_listbox.selection_clear(0, tk.END)
        self.chat_listbox.selection_set(idx)
        self._render_conversation()

    # ---- Renderowanie ----------------------------------------------------

    def _render_conversation(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        for msg in self.current_chat["messages"]:
            self._append_message(msg["role"], msg["content"], msg.get("attachments_note"))
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)

    def _append_message(self, role, content, attachments_note=None):
        self.text.configure(state=tk.NORMAL)
        label = "Ty" if role == "user" else "Model"
        tag = "user" if role == "user" else "assistant"
        self.text.insert(tk.END, f"{label}\n", tag)
        if attachments_note:
            self.text.insert(tk.END, f"📎 {attachments_note}\n", "attach")
        self.text.insert(tk.END, content + "\n", "body")
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)

    # ---- Wysyłanie -------------------------------------------------------

    def _on_enter(self, event):
        if event.state & 0x0001:  # Shift = nowa linia
            return
        self._send()
        return "break"

    def _send(self):
        if self.streaming:
            return
        text = self.entry.get("1.0", tk.END).strip()
        if not text and not self.pending_attachments:
            return

        slug = self._current_slug()
        if not slug:
            messagebox.showwarning("Brak workspace", "Najpierw połącz się i wybierz workspace.")
            return

        if not self.current_chat:
            self._new_chat()

        attachments = list(self.pending_attachments)
        note = None
        if attachments:
            note = "Załączono: " + ", ".join(a["name"] for a in attachments)

        self.entry.delete("1.0", tk.END)
        self._clear_attachments()

        # Wiadomość użytkownika.
        self.current_chat["messages"].append({
            "role": "user", "content": text, "attachments_note": note,
        })
        self._append_message("user", text, note)

        if self.current_chat["title"] == "Nowy czat":
            base = text or (note or "Dokument")
            self.current_chat["title"] = (base[:30] + "…") if len(base) > 30 else base
            self._refresh_chat_list()
            self._select_chat(self.current_chat["id"])
        self.store.save()

        # Nagłówek odpowiedzi.
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, "Model\n", "assistant")
        self.text.configure(state=tk.DISABLED)

        self._assistant_buffer = ""
        self.streaming = True
        self.send_btn.configure(state=tk.DISABLED)
        self.status_var.set("Model odpowiada…")

        threading.Thread(
            target=self._stream_worker, args=(slug, text, attachments), daemon=True
        ).start()

    def _stream_worker(self, slug, message, attachments):
        try:
            for chunk in self.client.stream_chat(slug, message, attachments):
                self.msg_queue.put(("chunk", chunk))
            self.msg_queue.put(("done", None))
        except urllib.error.HTTPError as e:
            self.msg_queue.put(("stream_error", f"HTTP {e.code}: {e.reason}"))
        except urllib.error.URLError as e:
            self.msg_queue.put(("stream_error", f"Błąd połączenia: {e.reason}"))
        except Exception as e:  # noqa: BLE001
            self.msg_queue.put(("stream_error", str(e)))

    # ---- Pętla zdarzeń z wątków -----------------------------------------

    def _process_queue(self):
        try:
            while True:
                kind, data = self.msg_queue.get_nowait()
                if kind == "workspaces":
                    self._on_workspaces(data)
                elif kind == "conn_error":
                    self.status_var.set(data)
                elif kind == "chunk":
                    self._on_chunk(data)
                elif kind == "done":
                    self._on_stream_done()
                elif kind == "stream_error":
                    self._on_stream_error(data)
        except queue.Empty:
            pass
        self.after(50, self._process_queue)

    def _on_workspaces(self, ws):
        self.workspaces = ws
        names = [n for n, s in ws]
        self.ws_combo.configure(values=names)
        if not names:
            self.status_var.set("Połączono, ale brak workspace. Utwórz jeden w AnythingLLM.")
            return
        # Przywróć zapisany wybór albo pierwszy.
        saved_slug = self.config_store.workspace_slug
        chosen = next((n for n, s in ws if s == saved_slug), names[0])
        self.ws_var.set(chosen)
        self._on_ws_selected()
        self.config_store.save()
        self.status_var.set(f"Połączono • {len(names)} workspace • model musi obsługiwać obraz, by czytać skany")

    def _on_chunk(self, chunk):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, chunk, "body")
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)
        self._assistant_buffer += chunk

    def _on_stream_done(self):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, "\n", "body")
        self.text.configure(state=tk.DISABLED)
        self.current_chat["messages"].append(
            {"role": "assistant", "content": self._assistant_buffer}
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
        self._apply_connection_fields()
        self.config_store.save()
        self.store.save()
        self.destroy()


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
