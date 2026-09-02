"""
czytaj_oswiadczenie.py — LOKALNY odczyt oświadczeń majątkowych (pismo ręczne).

Wersja na komputer z GPU (RTX 4060, 8 GB). Cała obróbka obrazu + rozpoznawanie
dzieje się LOKALNIE: model wizyjny chodzi na GPU przez Ollamę na localhost.
Nie potrzeba żadnego serwera w sieci.

Kontynuacja pomysłów z poprzedniej sesji (field_ocr.py), ale dwutorowo:

  1) PEŁNA TRANSKRYPCJA strony (dzieli stronę na poziome pasy w przerwach
     między wierszami -> pismo ręczne ma dużo wyższą rozdzielczość niż przy
     wysłaniu całej strony na raz). Działa na DOWOLNYM oświadczeniu majątkowym.

  2) EKSTRAKCJA PÓL wg szablonu fields_template.json (pole-po-polu) — najlepszy,
     ustrukturyzowany wynik dla formularza o znanym układzie (bryk.pdf).

Do obu dochodzi korekta słownikowa (RAG: 4 mln polskich słów, przywracanie
ogonków + ostrożny fuzzy z rapidfuzz) oraz heurystyczny licznik pewności.

Użycie:
    python czytaj_oswiadczenie.py bryk.pdf
        -> pełna transkrypcja + (dla bryk) ekstrakcja pól

    python czytaj_oswiadczenie.py Komorski_....pdf
        -> pełna transkrypcja dowolnego oświadczenia majątkowego

    python czytaj_oswiadczenie.py bryk.pdf --pola        # wymuś też ekstrakcję pól
    python czytaj_oswiadczenie.py bryk.pdf --kalibracja  # nakładki z ramkami pól
    python czytaj_oswiadczenie.py plik.pdf --model qwen2.5vl:7b --dpi 300

Wymaga: pymupdf, opencv-python, numpy, (opcjonalnie) rapidfuzz oraz działającej
lokalnej Ollamy z modelem wizyjnym (domyślnie qwen2.5vl:7b — pobierany
automatycznie, jeśli go brak).
"""

import argparse
import base64
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pymupdf

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# --- Konfiguracja lokalna -------------------------------------------------

BASE_DIR = Path(__file__).parent
TEMPLATE_FILE = BASE_DIR / "fields_template.json"
SLOWNIK_FILE = BASE_DIR / "slownik_polski.txt"

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"   # najlepszy pod pismo ręczne, mieści się w 8 GB
RENDER_DPI = 300                 # 300 to dobry kompromis jakość/VRAM na 8 GB

# Słownik urzędowy do korekty wartości słownikowych (typ "enum").
POLSKI_SLOWNIK = [
    "WŁASNOŚĆ", "WSPÓŁWŁASNOŚĆ", "MAŁŻEŃSKA WSPÓLNOŚĆ MAJĄTKOWA",
    "ODRĘBNA WŁASNOŚĆ", "DZIERŻAWA", "UŻYTKOWANIE WIECZYSTE", "NAJEM",
    "NIE DOTYCZY", "BRAK", "UŻYTKI ROLNE", "ZABUDOWA ZAGRODOWA", "DZIAŁKA",
    "LOKAL USŁUGOWY", "MIESZKANIE", "LAS", "POLE", "TAK", "NIE",
]


# --- Ollama: sprawdzenie/pobranie modelu ----------------------------------

def _api_get(path):
    with urllib.request.urlopen(f"{OLLAMA_URL}{path}", timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def ensure_ollama(model):
    """Sprawdza, czy Ollama działa i czy model jest pobrany. Jeśli nie ma
    modelu — pobiera go przez `ollama pull` (jednorazowo)."""
    try:
        tags = _api_get("/api/tags")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(
            "BŁĄD: nie widzę Ollamy na %s (%s).\n"
            "Uruchom serwer: `ollama serve` i spróbuj ponownie." % (OLLAMA_URL, e)
        )
    have = {m.get("name", "") for m in tags.get("models", [])}
    have |= {n.split(":")[0] for n in have}    # dopuść dopasowanie bez tagu
    if model in have or model.split(":")[0] in have:
        return
    print(f"Model '{model}' nie jest jeszcze pobrany — pobieram (jednorazowo)…")
    rc = subprocess.call(["ollama", "pull", model])
    if rc != 0:
        sys.exit(f"BŁĄD: `ollama pull {model}` zwrócił kod {rc}.")


# --- Przetwarzanie obrazu (lokalnie, CPU) --------------------------------

def render_page(pdf_path, page_index, dpi=RENDER_DPI):
    """Zwraca stronę PDF jako obraz BGR (numpy)."""
    doc = pymupdf.open(pdf_path)
    try:
        pix = doc.load_page(page_index).get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    finally:
        doc.close()


def page_count(pdf_path):
    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def deskew(img):
    """Delikatne prostowanie strony (koryguje niewielki obrót skanu)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thr)
    if coords is None:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.3 or abs(angle) > 15:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_gray(bgr):
    """Uwydatnia kontrast (CLAHE) i lekko wyostrza — bez zmiany rozmiaru."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def to_b64_png(gray, max_side=1600):
    """Skaluje do rozsądnego boku (VRAM) i koduje jako base64 PNG."""
    h, w = gray.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)),
                          interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", gray)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def find_band_cuts(gray, n_bands):
    """Dzieli stronę na ~n_bands poziomych pasów, tnąc w przerwach między
    wierszami (najmniej „atramentu"), żeby nie przeciąć tekstu w połowie."""
    h = gray.shape[0]
    if n_bands <= 1 or h < 400:
        return [(0, h)]
    ink = 255 - gray                                  # jasne tło -> ~0
    row_ink = ink.mean(axis=1)
    targets = [int(h * k / n_bands) for k in range(1, n_bands)]
    win = max(8, h // (n_bands * 12))                 # okno szukania przerwy
    cuts = [0]
    for t in targets:
        lo, hi = max(cuts[-1] + win, t - 4 * win), min(h - win, t + 4 * win)
        if hi <= lo:
            cuts.append(t)
            continue
        best = min(range(lo, hi), key=lambda r: row_ink[r - win:r + win].mean())
        cuts.append(best)
    cuts.append(h)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


# --- Zapytania do modelu --------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*?\|>")


def clean_text(text):
    """Sprząta odczyt: usuwa tokeny szablonowe, HTML, LaTeX, markdown."""
    t = _SPECIAL_TOKEN_RE.sub("", text or "")
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", t)
    t = t.replace("$", "").replace("`", "")
    t = re.sub(r"[#*_]{1,}", "", t)
    if t.strip().lower().startswith("assistant"):
        t = t.strip()[len("assistant"):]
    return collapse_repeats(t.strip())


def collapse_repeats(text, max_run=2):
    """Zwija pętle modelu: powtarzającą się linię (np. '[nieczytelne]' w kółko)
    ogranicza do max_run wystąpień; skleja też nadmiar pustych linii."""
    out, prev, run = [], None, 0
    for ln in (text or "").split("\n"):
        k = ln.strip()
        if not k:
            out.append("")
            continue
        if k == prev:
            run += 1
            if run >= max_run:
                continue
        else:
            prev, run = k, 0
        out.append(ln)
    joined = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    # Cały blok to tylko powtórzone [nieczytelne]/BRAK -> sprowadź do jednego.
    uniq = {l.strip().lower() for l in joined.split("\n") if l.strip()}
    if uniq and uniq <= {"[nieczytelne]", "brak", "[brak]"}:
        return "[nieczytelne]"
    return joined


def ask_vision(model, b64, prompt, num_ctx=8192, num_predict=1536, timeout=600):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,     # twardy limit — ucina pętle modelu
            "repeat_penalty": 1.15,         # łagodnie zniechęca do zapętlania fraz
            "repeat_last_n": 128,           # (za mocna kara zniekształca cyfry/kwoty)
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return clean_text(data.get("message", {}).get("content") or "")


PROMPT_TRANSKRYPCJA = (
    "To jest fragment polskiego dokumentu urzędowego (oświadczenie o stanie "
    "majątkowym). Przepisz DOKŁADNIE cały widoczny tekst — zarówno pismo "
    "ODRĘCZNE, jak i drukowane. Zachowaj kolejność i podział na wiersze. "
    "Nie tłumacz, nie streszczaj, nie interpretuj, nie dodawaj komentarzy "
    "ani nagłówków od siebie. Zachowaj polskie znaki (ą, ć, ę, ł, ń, ó, ś, ź, ż). "
    "Puste linie i pola pomiń. Jeśli czegoś nie da się odczytać, wpisz "
    "[nieczytelne]. Zwróć wyłącznie przepisany tekst."
)


# --- Korekta słownikowa (RAG) --------------------------------------------

_FOLD = str.maketrans("ĄĆĘŁŃÓŚŹŻ", "ACELNOSZZ")
_DICT_SET = None
_DICT_FOLDED = None
_FOLDED_BY_LEN = None


def _fold(s):
    return s.upper().translate(_FOLD)


def _load_dict():
    global _DICT_SET, _DICT_FOLDED, _FOLDED_BY_LEN
    if _DICT_SET is not None:
        return
    _DICT_SET, _DICT_FOLDED, _FOLDED_BY_LEN = set(), {}, {}
    if not SLOWNIK_FILE.exists():
        return
    for w in SLOWNIK_FILE.read_text(encoding="utf-8").split("\n"):
        w = w.strip()
        if len(w) < 3:
            continue
        _DICT_SET.add(w)
        fo = _fold(w)
        if fo not in _DICT_FOLDED:
            _DICT_FOLDED[fo] = w
            _FOLDED_BY_LEN.setdefault(len(fo), []).append(fo)


def _dopasuj_wielkosc(oryg, slowo):
    if oryg.isupper():
        return slowo
    if oryg[:1].isupper():
        return slowo.capitalize()
    return slowo.lower()


def _popraw_slowo(tok):
    up = tok.upper()
    if up in _DICT_SET:
        return tok
    fo = _fold(up)
    if fo in _DICT_FOLDED:                       # przywrócenie ogonków (bezpieczne)
        return _dopasuj_wielkosc(tok, _DICT_FOLDED[fo])
    if not HAS_RAPIDFUZZ:
        return tok
    cands = []
    for L in (len(fo) - 1, len(fo), len(fo) + 1):
        cands += _FOLDED_BY_LEN.get(L, [])
    if not cands:
        return tok
    hit = process.extractOne(fo, cands, scorer=fuzz.ratio)
    if hit and hit[1] >= 92 and len(hit[0]) == len(fo):
        return _dopasuj_wielkosc(tok, _DICT_FOLDED[hit[0]])
    return tok


def slownik_popraw(text):
    """Poprawia polskie słowa (len>=4): najpierw przywraca ogonki, potem
    ostrożny fuzzy. Nie rusza liczb, skrótów ani krótkich tokenów."""
    _load_dict()
    if not _DICT_SET:
        return text
    return re.sub(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}",
                  lambda m: _popraw_slowo(m.group(0)), text)


# --- Heurystyczny licznik pewności ---------------------------------------

def confidence_text(text):
    """Zgrubna pewność transkrypcji bloku (0-100) — bez logprobs.
    Karze pustkę, dużo [nieczytelne] i artefakty; nagradza sensowny udział
    liter/cyfr i długość."""
    t = (text or "").strip()
    if not t:
        return 0
    n_unreadable = t.lower().count("[nieczytelne]")
    core = re.sub(r"\[nieczytelne\]", "", t)
    letters = sum(c.isalnum() or c.isspace() for c in core)
    ratio = letters / max(1, len(core))
    score = 40 + int(ratio * 55)
    score -= min(40, n_unreadable * 8)
    if len(core.strip()) < 3:
        score -= 30
    if re.search(r"(.)\1{6,}", core):           # zapętlenie modelu
        score -= 25
    return max(0, min(100, score))


# --- TRYB 1: pełna transkrypcja (dowolne oświadczenie) --------------------

def bands_for_height(h):
    """Ile pasów na stronę — wyższe strony dzielimy na więcej części."""
    if h < 1600:
        return 1
    if h < 2600:
        return 2
    return 3


def transcribe(pdf_path, model, dpi):
    n_pages = page_count(pdf_path)
    print(f"Model: {model} • DPI: {dpi} • stron: {n_pages}")
    pages_out = []
    for pi in range(n_pages):
        print(f"\n=== Strona {pi + 1}/{n_pages} ===")
        gray = enhance_gray(deskew(render_page(pdf_path, pi, dpi)))
        h = gray.shape[0]
        nb = bands_for_height(h)
        cuts = find_band_cuts(gray, nb)
        block_texts, block_confs = [], []
        for bi, (y1, y2) in enumerate(cuts):
            band = gray[y1:y2]
            b64 = to_b64_png(band, max_side=1600)
            t0 = time.time()
            try:
                raw = ask_vision(model, b64, PROMPT_TRANSKRYPCJA, num_ctx=8192)
            except (urllib.error.URLError, OSError) as e:
                raw = f"[BŁĄD: {e}]"
            corrected = slownik_popraw(raw)
            conf = confidence_text(raw)
            block_texts.append(corrected)
            block_confs.append(conf)
            dt = time.time() - t0
            print(f"  pas {bi + 1}/{len(cuts)}  ({conf}%, {dt:.1f}s)")
        page_text = "\n".join(t for t in block_texts if t.strip())
        page_conf = int(np.mean(block_confs)) if block_confs else 0
        pages_out.append({"page": pi + 1, "confidence": page_conf,
                          "text": page_text})
    return pages_out


# --- TRYB 2: ekstrakcja pól wg szablonu (bryk) ---------------------------

TYPE_HINT = {
    "name": "To pole zawiera imię i nazwisko (zwykle wielkimi literami).",
    "date": "To pole zawiera datę (dzień, miesiąc, rok).",
    "place": "To pole zawiera nazwę miejscowości lub urzędu.",
    "amount": "To pole zawiera liczbę/kwotę (mogą być zł, gr, waluta).",
    "enum": "To pole zawiera krótki wpis urzędowy (np. WŁASNOŚĆ, NIE DOTYCZY).",
    "text": "",
}


def crop_norm(img, box):
    H, W = img.shape[:2]
    x, y, w, h = box
    x1, y1 = max(0, int(x * W)), max(0, int(y * H))
    x2, y2 = min(W, int((x + w) * W)), min(H, int((y + h) * H))
    return img[y1:y2, x1:x2]


def enhance_field(bgr):
    """Wycinek pola: CLAHE + powiększenie małych pól + wyostrzenie + ramka."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape
    if 0 < h < 200:
        s = min(4.0, 200 / h)
        gray = cv2.resize(gray, (int(w * s), int(h * s)),
                          interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    gray = cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", gray)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def recognize_field(model, b64, ftype):
    prompt = (
        "To jest powiększony wycinek JEDNEGO pola formularza. Odczytaj tylko to, "
        "co zostało w tym polu napisane (ręcznie lub drukiem). Zwróć samą wartość "
        "- bez komentarza, bez etykiety. " + TYPE_HINT.get(ftype, "") +
        " Jeśli pole jest puste, napisz dokładnie: PUSTE. "
        "Jeśli zupełnie nieczytelne: [nieczytelne]."
    )
    return ask_vision(model, b64, prompt, num_ctx=4096)


def _fuzzy_enum(value):
    if not HAS_RAPIDFUZZ:
        return value, 60
    hit = process.extractOne(value.upper(), POLSKI_SLOWNIK, scorer=fuzz.WRatio)
    if not hit:
        return value, 50
    term, score = hit[0], int(hit[1])
    if score >= 90:
        return term, min(88, score)
    if score >= 78:
        return f"{term}?  (odczyt: {value})", 60
    return value, 45


def validate(ftype, value):
    v = re.sub(r"\s+", " ", (value or "").strip())
    up = v.upper()
    if up == "PUSTE":
        return "", 100
    if not v or up == "[NIECZYTELNE]":
        return "[nieczytelne]", 25
    if ftype == "date":
        m = re.search(r"\d{1,2}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{2,4}", v)
        return v, (95 if m else 50)
    if ftype == "amount":
        return v, (90 if len(re.sub(r"\D", "", v)) >= 2 else 45)
    if ftype == "enum":
        return _fuzzy_enum(v)
    if ftype == "name":
        return v, (85 if len(v) >= 4 and " " in v else 65)
    return v, (80 if len(v) >= 3 else 50)


def extract_fields(pdf_path, model, dpi, out_json=None):
    template = json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
    results = []
    for page in template["pages"]:
        pi = page["page_index"]
        if pi >= page_count(pdf_path):
            break
        print(f"\n=== [pola] Strona {pi + 1} ===")
        img = deskew(render_page(pdf_path, pi, dpi))
        for f in page["fields"]:
            crop = crop_norm(img, f["box"])
            if crop.size == 0:
                continue
            try:
                raw = recognize_field(model, enhance_field(crop), f.get("type", "text"))
            except (urllib.error.URLError, OSError) as e:
                raw = f"[BŁĄD: {e}]"
            value, conf = validate(f.get("type", "text"), raw)
            if f.get("type") in ("text", "place"):
                value = slownik_popraw(value)
            flag = "" if conf >= 70 else "  <-- SPRAWDZ"
            print(f"  {f['label']:<34} : {value!r}  ({conf}%){flag}")
            results.append({"page": pi + 1, "id": f["id"], "label": f["label"],
                            "type": f.get("type", "text"), "value": value,
                            "raw": raw, "confidence": conf})
        if out_json:
            out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    return results


def calibrate(pdf_path, dpi):
    template = json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
    for page in template["pages"]:
        pi = page["page_index"]
        if pi >= page_count(pdf_path):
            break
        img = render_page(pdf_path, pi, dpi)
        H, W = img.shape[:2]
        for f in page["fields"]:
            x, y, w, h = f["box"]
            p1 = (int(x * W), int(y * H))
            p2 = (int((x + w) * W), int((y + h) * H))
            cv2.rectangle(img, p1, p2, (0, 0, 255), 3)
            cv2.putText(img, f["id"], (p1[0], max(0, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out = BASE_DIR / f"calib_str{pi + 1}.png"
        cv2.imwrite(str(out), img)
        print("Zapisano:", out)


# --- Zapis raportów -------------------------------------------------------

def save_transcription(stem, pages):
    out_json = BASE_DIR / f"{stem}_transkrypcja.json"
    out_txt = BASE_DIR / f"{stem}_transkrypcja.txt"
    out_json.write_text(json.dumps(pages, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    lines = []
    for p in pages:
        mark = "" if p["confidence"] >= 60 else "   [NISKA PEWNOŚĆ — SPRAWDŹ]"
        lines.append(f"\n===== STRONA {p['page']}  (pewność {p['confidence']}%){mark} =====")
        lines.append(p["text"] or "[pusto]")
    out_txt.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out_json, out_txt


def save_fields(stem, results):
    out_json = BASE_DIR / f"{stem}_pola.json"
    out_txt = BASE_DIR / f"{stem}_pola.txt"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    lines, cur = [], None
    for r in results:
        if r["page"] != cur:
            cur = r["page"]
            lines.append(f"\n===== STRONA {cur} =====")
        mark = "" if r["confidence"] >= 70 else "   [SPRAWDŹ]"
        lines.append(f"{r['label']}: {r['value']}{mark}")
    out_txt.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out_json, out_txt


# --- CLI ------------------------------------------------------------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="Lokalny odczyt oświadczeń majątkowych.")
    ap.add_argument("pdf", help="ścieżka do pliku PDF")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model wizyjny Ollamy")
    ap.add_argument("--dpi", type=int, default=RENDER_DPI, help="rozdzielczość renderu")
    ap.add_argument("--pola", action="store_true",
                    help="wymuś ekstrakcję pól wg szablonu (dla formularza bryk)")
    ap.add_argument("--tylko-pola", action="store_true",
                    help="tylko ekstrakcja pól, bez pełnej transkrypcji")
    ap.add_argument("--kalibracja", action="store_true",
                    help="zapisz nakładki z ramkami pól i zakończ")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        sys.exit(f"BŁĄD: nie ma pliku {pdf}")
    stem = pdf.stem

    if args.kalibracja:
        calibrate(str(pdf), args.dpi)
        return

    ensure_ollama(args.model)

    # Ekstrakcja pól: automatycznie dla 'bryk', albo gdy wymuszona flagą.
    do_fields = args.pola or args.tylko_pola or stem.lower().startswith("bryk")

    if not args.tylko_pola:
        pages = transcribe(str(pdf), args.model, args.dpi)
        j, t = save_transcription(stem, pages)
        print(f"\nZapisano transkrypcję: {j.name} oraz {t.name}")

    if do_fields:
        if not TEMPLATE_FILE.exists():
            print("(Pomijam pola — brak fields_template.json)")
        else:
            out_json = BASE_DIR / f"{stem}_pola.json"
            results = extract_fields(str(pdf), args.model, args.dpi, out_json=out_json)
            j, t = save_fields(stem, results)
            print(f"Zapisano pola: {j.name} oraz {t.name}")
    elif not args.tylko_pola:
        print("(Ten dokument nie ma szablonu pól — zrobiłem pełną transkrypcję. "
              "Wymuś ekstrakcję pól flagą --pola, jeśli układ pasuje do wzoru.)")


if __name__ == "__main__":
    main()
