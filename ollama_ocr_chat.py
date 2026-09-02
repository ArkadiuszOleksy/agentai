"""Ollama HTR - Enterprise Edition with Diagnostic Logs & ETA"""

import base64
from datetime import datetime
import io
import json
from pathlib import Path
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import urllib.error
import urllib.request

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pymupdf
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# --- KONFIGURACJA BAZOWA ---
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
PREFERRED_VISION_MODELS = ["qwen3-vl:8b", "minicpm-v:latest", "llama3.2-vision:latest"]
PREFERRED_TEXT_MODELS = ["llama3.1:8b", "qwen2.5:7b", "mistral:latest"]
DEFAULT_TEMPERATURE = 0.0

BASE_DPI = 300
RETRY_DPI = 450
CONFIDENCE_THRESHOLD = 70

# --- BAZA WIEDZY I PROMPTY ---
TRANSCRIBE_PROMPT = """Odczytaj wyłącznie wartości wpisane odręcznie w pola formularza. Zignoruj drukowane preambuły. Przepisz dokładnie to, co widzisz, uwzględniając polskie znaki."""

CORRECTION_PROMPT = """Jesteś ekspertem ds. polskich dokumentów prawnych. 
Poniżej znajduje się surowy tekst z systemu OCR, zawierający błędy z powodu nieczytelnego pisma odręcznego.
Twoje zadanie:
1. Popraw literówki, ucięte ogonki (ą, ć, ę, ł, ń, ó, ś, ź, ż) i zrekonstruuj nazwy urzędów/miejscowości.
2. Zastosuj się do formatu: "- **[Nazwa pola]**: [Odczytana wartość]"
3. Nie zmyślaj danych, których nie ma w tekście. Zwróć sam poprawiony tekst.

SUROWY TEKST DO KOREKTY:
{raw_text}"""

POLSKI_SLOWNIK_URZEDOWY = [
    "WŁASNOŚĆ", "WSPÓŁWŁASNOŚĆ", "MAŁŻEŃSKA WSPÓLNOŚĆ MAJĄTKOWA", "ODRĘBNA WŁASNOŚĆ",
    "DZIERŻAWA", "UŻYTKOWANIE WIECZYSTE", "NAJEM", "NIE DOTYCZY", "BRAK",
    "ROLA", "ZABUDOWA ZAGRODOWA", "DZIAŁKA", "LOKAL USŁUGOWY", "MIESZKANIE", "LAS",
    "ŚWIĘTOKRZYSKI URZĄD WOJEWÓDZKI", "URZĄD MIASTA", "URZĄD GMINY", "STAROSTWO POWIATOWE",
    "AGENCJA RESTRUKTURYZACJI I MODERNIZACJI ROLNICTWA", "ARiMR", "ZAKŁAD UBEZPIECZEŃ SPOŁECZNYCH", "ZUS",
    "DOCHÓD", "PRZYCHÓD", "ZOBOWIĄZANIA PIENIĘŻNE", "KREDYT HIPOTECZNY", "POŻYCZKA"
]
SPECIAL_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end|grounding|endoftext|pad)[^|]*?\|>")

# --- SILNIK WIZYJNY OpenCV (CPU) ---
def preprocess_with_opencv(image_bytes):
    if not HAS_CV2:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.8)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return [base64.b64encode(buffer.getvalue()).decode("ascii")]

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced_gray, 200, 255, cv2.THRESH_TRUNC)
    enhanced_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 0.4, enhanced_bgr, 0.6, 0)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 20
        H, W = img.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        cropped_img = img[y1:y2, x1:x2]
    else:
        cropped_img = img

    h_c, w_c = cropped_img.shape[:2]
    chunks = []
    if h_c > w_c * 1.1:
        step = h_c // 3
        overlap = int(h_c * 0.05)
        y_starts = [0, step - overlap, 2 * step - overlap]
        y_ends = [step + overlap, 2 * step + overlap, h_c]
        for ys, ye in zip(y_starts, y_ends):
            chunks.append(cropped_img[ys:ye, 0:w_c])
    else:
        chunks.append(cropped_img)

    b64_chunks = []
    for chunk in chunks:
        rgb_chunk = cv2.cvtColor(chunk, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_chunk)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=95)
        b64_chunks.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

    return b64_chunks

def slownikowa_korekta_htr(tekst):
    if not HAS_RAPIDFUZZ or not tekst: return tekst
    poprawiony = tekst
    slowa = [w for w in re.split(r'\W+', tekst) if len(w) > 5]
    for slowo in slowa:
        dopasowanie = process.extractOne(slowo.upper(), POLSKI_SLOWNIK_URZEDOWY, scorer=fuzz.WRatio)
        if dopasowanie and dopasowanie[1] > 88:
            poprawiony = re.sub(rf'\b{slowo}\b', dopasowanie[0], poprawiony, flags=re.IGNORECASE)
    return poprawiony

def clean_ocr_text(text):
    text = SPECIAL_TOKEN_RE.sub("", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?:None){2,}", "", text)
    text = text.lstrip()
    if text.lower().startswith("assistant\n"):
        text = text[len("assistant\n") :].lstrip()
    return text

def compute_confidence(text):
    t = (text or "").strip()
    if not t: return 0
    if len(t) < 20: return 10
    conf = 100.0
    if "nieczytelne" in t.lower(): conf -= min(40, t.lower().count("nieczytelne") * 10)
    if not any(c in set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ") for c in t): conf -= 15
    good = sum(ch.isalnum() or ch.isspace() or ch in set(",.;:%()/-–—+!?\"'“”„«»°²³ €$zł\n\r\t*-_:[]") for ch in t)
    ratio = good / len(t)
    if ratio < 0.85: conf -= (0.85 - ratio) * 200
    return max(5, min(100, round(conf)))

class OllamaClient:
    def __init__(self, base_url, logger_callback=None):
        self.base_url = base_url.rstrip("/")
        self.log = logger_callback

    def list_models(self):
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        out = []
        for m in data.get("models", []):
            caps = m.get("capabilities") or []
            out.append({"name": m["name"], "vision": ("vision" in caps) or any(k in m["name"].lower() for k in ["vl", "vision", "minicpm"])})
        return out

    def chat_stream(self, model, messages, temperature, num_ctx=32768):
        url = f"{self.base_url}/api/chat"
        payload = json.dumps({"model": model, "messages": messages, "stream": True, "options": {"temperature": float(temperature), "num_ctx": int(num_ctx)}}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        if self.log: self.log(f"[Sieć] Nawiązywanie połączenia z {url} (model: {model})...")
        start_net = time.time()

        with urllib.request.urlopen(req, timeout=600) as resp:
            first_chunk_received = False
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line: continue
                obj = json.loads(line)

                if not first_chunk_received:
                    if self.log: self.log(f"[Sieć] Serwer odpowiedział po {time.time()-start_net:.1f}s. Rozpoczęto pobieranie tokenów.")
                    first_chunk_received = True

                if obj.get("error"): raise RuntimeError(str(obj["error"]))
                if obj.get("message", {}).get("content"): yield obj["message"]["content"]
                if obj.get("done"): break

class Config:
    def __init__(self, path):
        self.path = path
        self.base_url = DEFAULT_BASE_URL
        self.vision_model = PREFERRED_VISION_MODELS[0]
        self.text_model = PREFERRED_TEXT_MODELS[0]
        self.load()

    def load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.base_url = d.get("base_url", self.base_url)
                self.vision_model = d.get("vision_model", self.vision_model)
                self.text_model = d.get("text_model", self.text_model)
            except: pass

    def save(self):
        try:
            self.path.write_text(json.dumps({"base_url": self.base_url, "vision_model": self.vision_model, "text_model": self.text_model}, ensure_ascii=False, indent=2), encoding="utf-8")
        except: pass

class HistoryStore:
    def __init__(self, path):
        self.path = path
        self.chats = []
        self.load()

    def load(self):
        if self.path.exists():
            try: self.chats = json.loads(self.path.read_text(encoding="utf-8"))
            except: self.chats = []

    def save(self):
        try:
            self.path.write_text(json.dumps(self.chats, ensure_ascii=False, indent=2), encoding="utf-8")
        except: pass

    def new_chat(self):
        chat = {"id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "title": "Nowy dokument", "created": datetime.now().isoformat(timespec="seconds"), "messages": []}
        self.chats.insert(0, chat)
        return chat

    def delete(self, chat_id):
        self.chats = [c for c in self.chats if c["id"] != chat_id]
        self.save()

class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ollama Enterprise HTR - Diagnostic Mode")
        self.geometry("1250x850")

        self.cfg = Config(Path(__file__).with_name("ollama_ocr_config.json"))
        self.client = OllamaClient(self.cfg.base_url, logger_callback=self._log_to_gui)
        self.store = HistoryStore(Path(__file__).with_name("ollama_ocr_history.json"))

        self.current_chat = None
        self.streaming = False
        self.msg_queue = queue.Queue()
        self.models = []
        self.pending_attachments = []
        self._pdf_store = {}
        self._assistant_buffer = ""

        self._build_ui()
        self._refresh_chat_list()
        self._load_models_async()

        if not self.store.chats: self._new_chat()
        else: self._select_chat(self.store.chats[0]["id"])

        self.after(50, self._process_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _log_to_gui(self, msg):
        """Wypisuje log do konsoli systemowej i wrzuca do kolejki GUI."""
        stamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{stamp}] {msg}"
        print(formatted)
        self.msg_queue.put(("log", formatted))

    def _build_ui(self):
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="URL:").pack(side=tk.LEFT)
        self.url_var = tk.StringVar(value=self.cfg.base_url)
        ttk.Entry(top, textvariable=self.url_var, width=20).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text="VLM:").pack(side=tk.LEFT)
        self.vis_model_var = tk.StringVar(value=self.cfg.vision_model)
        self.vis_combo = ttk.Combobox(top, textvariable=self.vis_model_var, width=18, state="readonly")
        self.vis_combo.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text="LLM:").pack(side=tk.LEFT)
        self.txt_model_var = tk.StringVar(value=self.cfg.text_model)
        self.txt_combo = ttk.Combobox(top, textvariable=self.txt_model_var, width=18, state="readonly")
        self.txt_combo.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(top, text="Odśwież", command=self._load_models_async).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Gotowy")
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT, padx=10)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = ttk.Frame(main, width=220)
        main.add(left, weight=0)
        btns = ttk.Frame(left)
        btns.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Button(btns, text="+ Nowy dokument", command=self._new_chat).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btns, text="Usuń", command=self._delete_chat, width=6).pack(side=tk.LEFT, padx=(4, 0))
        self.chat_listbox = tk.Listbox(left, activestyle="none", exportselection=False, font=("Segoe UI", 9))
        self.chat_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.chat_listbox.bind("<<ListboxSelect>>", self._on_chat_selected)

        right = ttk.Frame(main)
        main.add(right, weight=1)
        self.text = tk.Text(right, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10), padx=12, pady=10, background="#ffffff")
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        vscroll = ttk.Scrollbar(self.text, command=self.text.yview)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=vscroll.set)

        self.text.tag_configure("user", foreground="#0b5cad", font=("Segoe UI", 10, "bold"), spacing1=8)
        self.text.tag_configure("assistant", foreground="#1a7f37", font=("Segoe UI", 10, "bold"), spacing1=8)
        self.text.tag_configure("log_tag", foreground="#888888", font=("Consolas", 9))
        self.text.tag_configure("body", foreground="#111111", spacing2=2, spacing3=4, lmargin1=4, lmargin2=4)
        self.text.tag_configure("attach", foreground="#8a5a00", font=("Segoe UI", 9))
        self.text.tag_configure("page", foreground="#444444", font=("Segoe UI", 10, "bold"), spacing1=10, spacing3=4)
        self.text.tag_configure("confok", foreground="#1a7f37", font=("Consolas", 9, "bold"))

        attach_bar = ttk.Frame(right, padding=(0, 4))
        attach_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(attach_bar, text="📎 Wybierz pliki", command=self._attach_files).pack(side=tk.LEFT)
        self.attach_label = ttk.Label(attach_bar, text="Brak załączników", foreground="#777")
        self.attach_label.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(attach_bar, text="Wyczyść", command=self._clear_attachments).pack(side=tk.RIGHT)

        bottom = ttk.Frame(right, padding=(0, 4))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.entry = tk.Text(bottom, height=3, wrap=tk.WORD, font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.send_btn = ttk.Button(bottom, text="URUCHOM POTOK HTR\n(Z DIAGNOSTYKĄ)", command=self._send)
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))

    def _load_models_async(self):
        self.client.base_url = self.url_var.get().strip()
        self._log_to_gui("Pobieranie listy modeli z serwera...")
        threading.Thread(target=lambda: self.msg_queue.put(("models", self.client.list_models())), daemon=True).start()

    def _attach_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("Dokumenty", "*.png *.jpg *.jpeg *.pdf")])
        for p in paths:
            path = Path(p)
            if path.suffix.lower() == ".pdf" and HAS_PDF:
                pdf_bytes = path.read_bytes()
                pdf_id = f"pdf_{len(self._pdf_store)}_{path.name}"
                self._pdf_store[pdf_id] = pdf_bytes
                with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
                    for i in range(doc.page_count):
                        self.pending_attachments.append({"name": f"{path.name} (str {i+1})", "kind": "pdf", "pdf_id": pdf_id, "page_index": i})
            else:
                self.pending_attachments.append({"name": path.name, "kind": "image", "data": path.read_bytes()})
        self.attach_label.configure(text=f"Dołączono stron: {len(self.pending_attachments)}")

    def _clear_attachments(self):
        self.pending_attachments = []
        self.attach_label.configure(text="Brak załączników")

    def _refresh_chat_list(self):
        self.chat_listbox.delete(0, tk.END)
        for chat in self.store.chats: self.chat_listbox.insert(tk.END, f"  {chat['title']}")

    def _new_chat(self):
        chat = self.store.new_chat()
        self.store.save()
        self._refresh_chat_list()
        self._select_chat(chat["id"])

    def _delete_chat(self):
        if self.current_chat and messagebox.askyesno("Usuń", "Usunąć z historii?"):
            self.store.delete(self.current_chat["id"])
            self.current_chat = None
            self._refresh_chat_list()
            self._select_chat(self.store.chats[0]["id"] if self.store.chats else self._new_chat()["id"])

    def _on_chat_selected(self, _event):
        sel = self.chat_listbox.curselection()
        if sel: self._select_chat(self.store.chats[sel[0]]["id"])

    def _select_chat(self, chat_id):
        self.current_chat = next((c for c in self.store.chats if c["id"] == chat_id), None)
        if self.current_chat:
            self.chat_listbox.selection_clear(0, tk.END)
            self.chat_listbox.selection_set(self.store.chats.index(self.current_chat))
            self.text.configure(state=tk.NORMAL)
            self.text.delete("1.0", tk.END)
            for msg in self.current_chat["messages"]:
                tag = "user" if msg["role"] == "user" else "assistant"
                self.text.insert(tk.END, f"{'Użytkownik' if tag=='user' else 'System'}:\n", tag)
                if msg.get("attachments_note"): self.text.insert(tk.END, f"📎 {msg['attachments_note']}\n", "attach")
                self.text.insert(tk.END, msg["content"] + "\n\n", "body")
            self.text.configure(state=tk.DISABLED)
            self.text.see(tk.END)

    def _on_enter(self, event):
        if not (event.state & 0x0001):
            self._send()
            return "break"

    def _send(self):
        if self.streaming or not self.pending_attachments: return
        self.streaming = True
        self.send_btn.configure(state=tk.DISABLED)

        attachments = list(self.pending_attachments)
        self.pending_attachments.clear()
        self.attach_label.configure(text="Brak dokumentów")

        note = f"Rozpoczęto analizę {len(attachments)} stron(y)..."
        self.current_chat["messages"].append({"role": "user", "content": "Analiza dokumentu.", "attachments_note": note})
        if self.current_chat["title"] == "Nowy dokument":
            self.current_chat["title"] = attachments[0]["name"]
            self._refresh_chat_list()

        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, f"Użytkownik:\n", "user")
        self.text.insert(tk.END, f"📎 {note}\n\n", "attach")
        self.text.configure(state=tk.DISABLED)

        self._assistant_buffer = ""
        threading.Thread(target=self._pipeline_worker, args=(self.vis_model_var.get(), self.txt_model_var.get(), attachments), daemon=True).start()

    def _pipeline_worker(self, v_model, t_model, pages):
        start_pipeline = time.time()

        for idx, page in enumerate(pages):
            page_start_time = time.time()
            self.msg_queue.put(("page_start", (idx + 1, len(pages), page["name"])))

            # --- FAZA 1: OpenCV ---
            self._log_to_gui(f"[Faza 1/4] Przygotowanie skanu (OpenCV CPU)...")
            cv2_start = time.time()
            try:
                if page["kind"] == "image": raw_bytes = page["data"]
                else:
                    with pymupdf.open(stream=self._pdf_store[page["pdf_id"]], filetype="pdf") as doc:
                        raw_bytes = doc.load_page(page["page_index"]).get_pixmap(dpi=BASE_DPI).tobytes("png")
                b64_images = preprocess_with_opencv(raw_bytes)
                self._log_to_gui(f"[Faza 1/4] Zakończono pomyślnie w {time.time()-cv2_start:.1f}s. Utworzono {len(b64_images)} wycinki obrazu.")
            except Exception as e:
                self._log_to_gui(f"[BŁĄD Fazy 1] OpenCV zawiódł: {e}")
                continue

            # --- FAZA 2: VLM ---
            self._log_to_gui(f"[Faza 2/4] Ekstrakcja tekstu VLM (Model: {v_model})...")
            vlm_start = time.time()
            raw_ocr = ""
            try:
                for chunk in self.client.chat_stream(v_model, [{"role": "user", "content": TRANSCRIBE_PROMPT, "images": b64_images}], DEFAULT_TEMPERATURE):
                    raw_ocr += chunk
                self._log_to_gui(f"[Faza 2/4] Odczyt VLM zakończony w {time.time()-vlm_start:.1f}s.")
            except Exception as e:
                self._log_to_gui(f"[BŁĄD Fazy 2] Awaria Ollamy: {e}")

            raw_ocr = clean_ocr_text(raw_ocr)

            # --- FAZA 3: RapidFuzz ---
            self._log_to_gui(f"[Faza 3/4] Słownikowa weryfikacja RAG...")
            fuzzed_start = time.time()
            fuzzed_text = slownikowa_korekta_htr(raw_ocr)
            self._log_to_gui(f"[Faza 3/4] Zakończono w {time.time()-fuzzed_start:.2f}s.")

            # --- FAZA 4: LLM ---
            self._log_to_gui(f"[Faza 4/4] Kontekstowe wygładzanie (Model: {t_model})...")
            llm_start = time.time()
            final_text = ""
            prompt = CORRECTION_PROMPT.format(raw_text=fuzzed_text)
            try:
                for chunk in self.client.chat_stream(t_model, [{"role": "user", "content": prompt}], 0.1, num_ctx=8192):
                    final_text += chunk
                    self.msg_queue.put(("chunk", chunk, "body"))
                self._log_to_gui(f"[Faza 4/4] LLM zakończył w {time.time()-llm_start:.1f}s.")
            except Exception as e:
                self._log_to_gui(f"[BŁĄD Fazy 4] Awaria LLM: {e}")
                final_text = fuzzed_text

            final_text = clean_ocr_text(final_text)
            conf = compute_confidence(final_text)
            self.msg_queue.put(("page_end", (conf, final_text)))

            # --- ETA CALCULATION ---
            pages_left = len(pages) - (idx + 1)
            if pages_left > 0:
                elapsed = time.time() - start_pipeline
                avg_time = elapsed / (idx + 1)
                eta_sec = int(avg_time * pages_left)
                m, s = divmod(eta_sec, 60)
                self._log_to_gui(f"--- Ukończono stronę {idx+1}. Szacowany czas do końca: {m}m {s}s ---")

        self.msg_queue.put(("done", "Zakończono potok HTR."))

    def _process_queue(self):
        try:
            while True:
                kind, data = self.msg_queue.get_nowait()
                if kind == "log":
                    self.text.configure(state=tk.NORMAL)
                    self.text.insert(tk.END, f"{data}\n", "log_tag")
                    self.text.configure(state=tk.DISABLED)
                    self.text.see(tk.END)
                elif kind == "models":
                    v_names = [m["name"] for m in data if m["vision"]]
                    t_names = [m["name"] for m in data if not m["vision"]] or [m["name"] for m in data]
                    self.vis_combo.configure(values=v_names)
                    self.txt_combo.configure(values=t_names)
                    if v_names: self.vis_model_var.set(next((p for p in PREFERRED_VISION_MODELS if p in v_names), v_names[0]))
                    if t_names: self.txt_model_var.set(next((p for p in PREFERRED_TEXT_MODELS if p in t_names), t_names[0]))
                elif kind == "page_start":
                    idx, tot, name = data
                    self.text.configure(state=tk.NORMAL)
                    self.text.insert(tk.END, f"\n--- WYNIK STRONA {idx}/{tot}: {name} ---\n", "page")
                    self.text.configure(state=tk.DISABLED)
                    self.text.see(tk.END)
                    self.status_var.set(f"Analiza strony {idx} z {tot}...")
                elif kind == "chunk":
                    chunk_text, tag = data[0], data[1] if len(data)>1 else "body"
                    self.text.configure(state=tk.NORMAL)
                    self.text.insert(tk.END, chunk_text, tag)
                    self.text.configure(state=tk.DISABLED)
                    self.text.see(tk.END)
                elif kind == "page_end":
                    conf, final_text = data
                    self.text.configure(state=tk.NORMAL)
                    self.text.insert(tk.END, f"\n\n[Pewność HTR: {conf}%]\n", "confok")
                    self.text.configure(state=tk.DISABLED)
                    self._assistant_buffer += f"\n--- Strona ---\n{final_text}\n"
                elif kind == "done":
                    self.current_chat["messages"].append({"role": "assistant", "content": self._assistant_buffer})
                    self.store.save()
                    self.streaming = False
                    self.send_btn.configure(state=tk.NORMAL)
                    self.status_var.set(data)
        except queue.Empty: pass
        self.after(50, self._process_queue)

    def _on_close(self):
        self.cfg.base_url = self.url_var.get().strip()
        self.cfg.vision_model = self.vis_model_var.get()
        self.cfg.text_model = self.txt_model_var.get()
        self.cfg.save()
        self.store.save()
        self.destroy()

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()