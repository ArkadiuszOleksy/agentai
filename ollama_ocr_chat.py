"""
Ollama OCR Chat - agent z GUI do przepisywania dokumentów 1:1.

Łączy się bezpośrednio z serwerem Ollama, pozwala DOŁĄCZAĆ zdjęcia/skany
(również pisma ręcznego) i prosić model vision o przepisanie treści 1:1.
Temperatura domyślnie 0 - minimalizuje halucynacje przy przepisywaniu.

Historia wielu czatów zapisywana lokalnie w pliku JSON.
Wymaga tylko biblioteki standardowej Pythona (tkinter + urllib).

Do odczytu obrazów model MUSI mieć zdolność 'vision'
(np. qwen3-vl, llama3.2-vision). Modele tekstowe obrazu nie zobaczą.
"""

import base64
import json
import queue
import threading
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Obsługa PDF (opcjonalna) - renderuje strony skanów na obrazy.
try:
    import pymupdf  # PyMuPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# --- Konfiguracja domyślna -----------------------------------------------

DEFAULT_BASE_URL = "http://192.168.100.53:11434"
PREFERRED_MODEL = "qwen3-vl:8b"          # najlepszy do pisma ręcznego
DEFAULT_TEMPERATURE = 0.0                # 0 = najmniej halucynacji

CONFIG_FILE = Path(__file__).with_name("ollama_ocr_config.json")
HISTORY_FILE = Path(__file__).with_name("ollama_ocr_history.json")
REQUEST_TIMEOUT = 600

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
PDF_RENDER_DPI = 200  # rozdzielczość renderowania stron PDF (skany pism)

# Dobór okna kontekstu (num_ctx). Obrazy zużywają dużo tokenów:
# ~2600 tok. na stronę (wejście) + zapas na przepisany tekst (wyjście).
TOKENS_PER_IMAGE = 2600
OUTPUT_TOKENS_PER_IMAGE = 2000
CTX_BASE = 8192        # domyślne okno bez obrazów
CTX_MIN = 8192
CTX_CAP = 65536        # górny limit (ochrona pamięci serwera)


def estimate_num_ctx(n_images):
    """Szacuje potrzebne num_ctx na podstawie liczby dołączonych obrazów."""
    if n_images <= 0:
        return CTX_BASE
    est = 2000 + n_images * (TOKENS_PER_IMAGE + OUTPUT_TOKENS_PER_IMAGE)
    rounded = ((est + 4095) // 4096) * 4096
    return max(CTX_MIN, min(CTX_CAP, rounded))

# Polecenie wstawiane automatycznie po dołączeniu obrazu.
TRANSCRIBE_PROMPT = (
    "Przepisz dokładnie 1:1 całą treść z załączonego dokumentu. "
    "Odwzoruj zarówno tekst drukowany, jak i pismo ręczne. "
    "Nie streszczaj, nie interpretuj, nie poprawiaj i nie tłumacz. "
    "Zachowaj oryginalny układ, interpunkcję, wielkość liter i podział "
    "na akapity. Jeśli fragment jest nieczytelny, oznacz go jako "
    "[nieczytelne] zamiast zgadywać."
)


# --- Klient Ollama --------------------------------------------------------

class OllamaClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def list_models(self):
        """Zwraca listę słowników: {'name', 'vision': bool, 'caps': [...]}."""
        url = f"{self.base_url}/api/tags"
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        out = []
        for m in data.get("models", []):
            caps = m.get("capabilities") or []
            out.append({
                "name": m["name"],
                "vision": "vision" in caps,
                "caps": caps,
            })
        return out

    def chat_stream(self, model, messages, temperature, num_ctx):
        """
        Strumieniowa rozmowa. Generator zwraca fragmenty odpowiedzi.
        messages: lista {'role','content','images'?} gdzie images to lista
        czystego base64 (bez prefiksu data:).
        num_ctx: rozmiar okna kontekstu (obrazy zużywają dużo tokenów).
        """
        url = f"{self.base_url}/api/chat"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {"temperature": float(temperature), "num_ctx": int(num_ctx)},
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise RuntimeError(str(obj["error"]))
                msg = obj.get("message") or {}
                if msg.get("content"):
                    yield msg["content"]
                if obj.get("done"):
                    break


# --- Ustawienia i historia -----------------------------------------------

class Config:
    def __init__(self, path):
        self.path = path
        self.base_url = DEFAULT_BASE_URL
        self.model = PREFERRED_MODEL
        self.temperature = DEFAULT_TEMPERATURE
        self.load()

    def load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.base_url = d.get("base_url", self.base_url)
                self.model = d.get("model", self.model)
                self.temperature = d.get("temperature", self.temperature)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        try:
            self.path.write_text(json.dumps({
                "base_url": self.base_url,
                "model": self.model,
                "temperature": self.temperature,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
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
        self.title("Ollama OCR Chat - przepisywanie dokumentów 1:1")
        self.geometry("1060x700")
        self.minsize(860, 540)

        self.cfg = Config(CONFIG_FILE)
        self.client = OllamaClient(self.cfg.base_url)
        self.store = HistoryStore(HISTORY_FILE)
        self.current_chat = None
        self.streaming = False
        self.msg_queue = queue.Queue()
        self.models = []               # [{'name','vision','caps'}]
        self.pending_attachments = []  # [{'name','b64','is_image'}]
        self._assistant_buffer = ""

        self._build_ui()
        self._refresh_chat_list()
        self._load_models_async()

        if not self.store.chats:
            self._new_chat()
        else:
            self._select_chat(self.store.chats[0]["id"])

        self.after(50, self._process_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- Layout ----------------------------------------------------------

    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Serwer:").pack(side=tk.LEFT)
        self.url_var = tk.StringVar(value=self.cfg.base_url)
        ttk.Entry(top, textvariable=self.url_var, width=24).pack(side=tk.LEFT, padx=(4, 10))

        ttk.Label(top, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.cfg.model)
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, width=26, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=(4, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        ttk.Label(top, text="Temperatura:").pack(side=tk.LEFT)
        self.temp_var = tk.DoubleVar(value=self.cfg.temperature)
        ttk.Spinbox(top, from_=0.0, to=1.0, increment=0.1, width=5,
                    textvariable=self.temp_var, command=self._on_temp_changed).pack(side=tk.LEFT, padx=(4, 10))

        ttk.Button(top, text="Połącz / Odśwież", command=self._load_models_async).pack(side=tk.LEFT)

        status_bar = ttk.Frame(self, padding=(8, 0))
        status_bar.pack(side=tk.TOP, fill=tk.X)
        self.status_var = tk.StringVar(value="Łączenie…")
        ttk.Label(status_bar, textvariable=self.status_var, foreground="#666").pack(side=tk.LEFT)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main, width=220)
        main.add(left, weight=0)
        btns = ttk.Frame(left)
        btns.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Button(btns, text="+ Nowy czat", command=self._new_chat).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btns, text="Usuń", command=self._delete_chat, width=6).pack(side=tk.LEFT, padx=(4, 0))
        self.chat_listbox = tk.Listbox(left, activestyle="none", exportselection=False)
        self.chat_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.chat_listbox.bind("<<ListboxSelect>>", self._on_chat_selected)

        right = ttk.Frame(main)
        main.add(right, weight=1)
        self.text = tk.Text(right, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10),
                            padx=10, pady=8, background="#fafafa")
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        vscroll = ttk.Scrollbar(self.text, command=self.text.yview)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=vscroll.set)
        self.text.tag_configure("user", foreground="#0b5cad", font=("Segoe UI", 10, "bold"), spacing1=8)
        self.text.tag_configure("assistant", foreground="#1a7f37", font=("Segoe UI", 10, "bold"), spacing1=8)
        self.text.tag_configure("body", foreground="#222", spacing3=4, lmargin1=4, lmargin2=4)
        self.text.tag_configure("attach", foreground="#8a5a00")
        self.text.tag_configure("error", foreground="#b00020")

        attach_bar = ttk.Frame(right)
        attach_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 2))
        ttk.Button(attach_bar, text="📎 Dołącz dokument", command=self._attach_files).pack(side=tk.LEFT)
        self.attach_label = ttk.Label(attach_bar, text="Brak załączników", foreground="#888")
        self.attach_label.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(attach_bar, text="Wyczyść", command=self._clear_attachments).pack(side=tk.RIGHT)

        bottom = ttk.Frame(right)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.entry = tk.Text(bottom, height=3, wrap=tk.WORD, font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.send_btn = ttk.Button(bottom, text="Wyślij", command=self._send)
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))

    # ---- Modele ----------------------------------------------------------

    def _load_models_async(self):
        self.cfg.base_url = self.url_var.get().strip()
        self.client = OllamaClient(self.cfg.base_url)
        self.status_var.set("Łączenie…")

        def worker():
            try:
                models = self.client.list_models()
                self.msg_queue.put(("models", models))
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.msg_queue.put(("models_error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _selected_model_info(self):
        name = self.model_var.get()
        return next((m for m in self.models if m["name"] == name), None)

    def _on_model_selected(self, _e=None):
        self.cfg.model = self.model_var.get()
        self.cfg.save()
        self._update_status_for_model()

    def _on_temp_changed(self):
        try:
            self.cfg.temperature = float(self.temp_var.get())
            self.cfg.save()
        except (tk.TclError, ValueError):
            pass

    def _update_status_for_model(self):
        info = self._selected_model_info()
        if not info:
            return
        if info["vision"]:
            self.status_var.set(f"Model {info['name']} • 👁 vision • gotowy do OCR")
        else:
            self.status_var.set(
                f"⚠ Model {info['name']} NIE obsługuje obrazów - nie odczyta zdjęć, "
                f"tylko tekst. Wybierz model z 👁 vision."
            )

    # ---- Załączniki ------------------------------------------------------

    def _attach_files(self):
        pdf_hint = " *.pdf" if HAS_PDF else ""
        paths = filedialog.askopenfilenames(
            title="Wybierz skany/PDF do przepisania",
            filetypes=[
                ("Skany i PDF", f"*.png *.jpg *.jpeg *.webp *.gif *.bmp *.tiff{pdf_hint}"),
                ("Obrazy", "*.png *.jpg *.jpeg *.webp *.gif *.bmp *.tiff"),
                ("PDF", "*.pdf"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not paths:
            return
        for p in paths:
            path = Path(p)
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                self._attach_pdf(path)
            elif suffix in IMAGE_EXTS:
                try:
                    raw = path.read_bytes()
                except OSError as e:
                    messagebox.showerror("Błąd", f"Nie można wczytać {path.name}: {e}")
                    continue
                self.pending_attachments.append({
                    "name": path.name,
                    "b64": base64.b64encode(raw).decode("ascii"),
                })
            else:
                messagebox.showwarning(
                    "Nieobsługiwany plik",
                    f"{path.name}: obsługiwane są obrazy (PNG/JPG…) oraz PDF.",
                )
        self._update_attach_label()

        if self.pending_attachments and not self.entry.get("1.0", tk.END).strip():
            self.entry.insert("1.0", TRANSCRIBE_PROMPT)

    def _attach_pdf(self, path):
        """Renderuje każdą stronę PDF do obrazu PNG i dodaje jako załącznik."""
        if not HAS_PDF:
            messagebox.showerror(
                "Brak obsługi PDF",
                "Aby wczytywać PDF, zainstaluj bibliotekę PyMuPDF:\n\n"
                "    pip install pymupdf",
            )
            return
        try:
            doc = pymupdf.open(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Błąd PDF", f"Nie można otworzyć {path.name}: {e}")
            return

        n_pages = doc.page_count
        if n_pages > 15 and not messagebox.askyesno(
            "Duży PDF",
            f"{path.name} ma {n_pages} stron. Każda strona to osobny obraz - "
            f"przepisywanie może długo trwać i mocno obciążyć model.\n\n"
            f"Kontynuować?",
        ):
            doc.close()
            return

        try:
            for i in range(n_pages):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
                png_bytes = pix.tobytes("png")
                self.pending_attachments.append({
                    "name": f"{path.name} [str. {i + 1}]",
                    "b64": base64.b64encode(png_bytes).decode("ascii"),
                })
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Błąd PDF", f"Błąd renderowania {path.name}: {e}")
        finally:
            doc.close()

    def _clear_attachments(self):
        self.pending_attachments = []
        self._update_attach_label()

    def _update_attach_label(self):
        n = len(self.pending_attachments)
        if n == 0:
            self.attach_label.configure(text="Brak załączników", foreground="#888")
        else:
            names = ", ".join(a["name"] for a in self.pending_attachments)
            self.attach_label.configure(text=f"{n} obraz(y): {names}", foreground="#8a5a00")

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

        info = self._selected_model_info()
        if self.pending_attachments and info and not info["vision"]:
            if not messagebox.askyesno(
                "Model bez obsługi obrazu",
                f"Model '{info['name']}' nie obsługuje obrazów - nie odczyta "
                f"załączników i może zmyślić treść.\n\nWysłać mimo to?",
            ):
                return

        # Sprawdź, czy liczba stron nie przekracza rozsądnego okna kontekstu.
        n_imgs = len(self.pending_attachments)
        needed = 2000 + n_imgs * (TOKENS_PER_IMAGE + OUTPUT_TOKENS_PER_IMAGE)
        if n_imgs and needed > CTX_CAP:
            max_pages = (CTX_CAP - 2000) // (TOKENS_PER_IMAGE + OUTPUT_TOKENS_PER_IMAGE)
            if not messagebox.askyesno(
                "Za dużo stron naraz",
                f"Dołączono {n_imgs} stron. Bezpiecznie mieści się ok. {max_pages} "
                f"stron na jedno zapytanie - przy większej liczbie model może "
                f"obciąć tekst lub zwrócić błąd.\n\nLepiej wyślij mniej stron naraz.\n\n"
                f"Wysłać mimo to?",
            ):
                return

        if not self.current_chat:
            self._new_chat()

        images_b64 = [a["b64"] for a in self.pending_attachments]
        note = None
        if self.pending_attachments:
            note = "Załączono: " + ", ".join(a["name"] for a in self.pending_attachments)

        self.entry.delete("1.0", tk.END)
        self._clear_attachments()

        # Zapis wiadomości użytkownika (obrazy tylko w bieżącym zapytaniu).
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

        # Budowa wiadomości dla API (historia jako tekst + obrazy w ostatniej).
        api_messages = []
        for m in self.current_chat["messages"]:
            api_messages.append({"role": m["role"], "content": m["content"]})
        if images_b64:
            api_messages[-1]["images"] = images_b64

        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, "Model\n", "assistant")
        self.text.configure(state=tk.DISABLED)

        self._assistant_buffer = ""
        self.streaming = True
        self.send_btn.configure(state=tk.DISABLED)
        self.status_var.set("Model przepisuje…")

        model = self.model_var.get()
        temp = self.temp_var.get()
        num_ctx = estimate_num_ctx(len(images_b64))
        threading.Thread(
            target=self._stream_worker,
            args=(model, api_messages, temp, num_ctx),
            daemon=True,
        ).start()

    def _stream_worker(self, model, messages, temperature, num_ctx):
        try:
            for chunk in self.client.chat_stream(model, messages, temperature, num_ctx):
                self.msg_queue.put(("chunk", chunk))
            self.msg_queue.put(("done", None))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:500]
            except Exception:  # noqa: BLE001
                detail = e.reason
            self.msg_queue.put(("stream_error", f"HTTP {e.code}: {detail}"))
        except urllib.error.URLError as e:
            self.msg_queue.put(("stream_error", f"Błąd połączenia: {e.reason}"))
        except Exception as e:  # noqa: BLE001
            self.msg_queue.put(("stream_error", str(e)))

    # ---- Pętla zdarzeń ---------------------------------------------------

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
        self.models = models
        # Etykiety z oznaczeniem vision.
        labels = [(m["name"] + ("  👁" if m["vision"] else "")) for m in models]
        self._label_to_name = {lbl: m["name"] for lbl, m in zip(labels, models)}
        self.model_combo.configure(values=[m["name"] for m in models])
        names = [m["name"] for m in models]
        if not names:
            self.status_var.set("Serwer nie zwrócił modeli")
            return
        # Wybór: zapamiętany > preferowany > pierwszy vision > pierwszy.
        chosen = self.cfg.model if self.cfg.model in names else None
        if not chosen and PREFERRED_MODEL in names:
            chosen = PREFERRED_MODEL
        if not chosen:
            chosen = next((m["name"] for m in models if m["vision"]), names[0])
        self.model_var.set(chosen)
        self._on_model_selected()

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
        self._update_status_for_model()
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
        self.cfg.base_url = self.url_var.get().strip()
        self.cfg.model = self.model_var.get()
        try:
            self.cfg.temperature = float(self.temp_var.get())
        except (tk.TclError, ValueError):
            pass
        self.cfg.save()
        self.store.save()
        self.destroy()


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
