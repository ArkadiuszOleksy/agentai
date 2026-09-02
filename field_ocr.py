"""
Field-by-field OCR dla formularzy o stałym układzie (np. oświadczenie
majątkowe, wzór Dz.U. poz. 1825).

Pomysł: zamiast wysyłać całą stronę do modelu (co daje błędy), tniemy
KAŻDE POLE osobno wg szablonu i rozpoznajemy pojedynczo. Małe wycinki =
dużo wyższa dokładność pisma odręcznego.

Sprzęt: całe cięcie/obróbka obrazu dzieje się LOKALNIE na CPU (tanie),
a rozpoznawanie robi model na serwerze Ollama (przez API).

Użycie:
    python field_ocr.py kalibracja bryk.pdf         # nakładki z ramkami
    python field_ocr.py rozpoznaj  bryk.pdf          # ekstrakcja pól -> JSON+TXT

Wymaga: pymupdf, opencv-python, numpy, pillow, (opcjonalnie) rapidfuzz.
"""

import base64
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pymupdf

try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# --- Konfiguracja ---------------------------------------------------------

BASE_DIR = Path(__file__).parent
TEMPLATE_FILE = BASE_DIR / "fields_template.json"
CONFIG_FILE = BASE_DIR / "ollama_ocr_config.json"   # współdzielone z aplikacją

DEFAULT_URL = "http://192.168.100.53:11434"
DEFAULT_MODEL = "qwen3-vl:8b"
RENDER_DPI = 500
FIELD_NUM_CTX = 50000

# Słownik urzędowy do korekty wartości słownikowych (typ "enum").
POLSKI_SLOWNIK = [
    "WŁASNOŚĆ", "WSPÓŁWŁASNOŚĆ", "MAŁŻEŃSKA WSPÓLNOŚĆ MAJĄTKOWA",
    "ODRĘBNA WŁASNOŚĆ", "DZIERŻAWA", "UŻYTKOWANIE WIECZYSTE", "NAJEM",
    "NIE DOTYCZY", "BRAK", "UŻYTKI ROLNE", "ZABUDOWA ZAGRODOWA", "DZIAŁKA",
    "LOKAL USŁUGOWY", "MIESZKANIE", "LAS", "POLE",
]


def load_runtime():
    """URL serwera i model - z pliku konfiguracyjnego aplikacji, jeśli jest."""
    url, model = DEFAULT_URL, DEFAULT_MODEL
    if CONFIG_FILE.exists():
        try:
            d = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            url = d.get("base_url", url)
            model = d.get("vision_model", d.get("model", model))
        except (json.JSONDecodeError, OSError):
            pass
    return url.rstrip("/"), model


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
        return img  # nic albo podejrzanie dużo - nie ruszaj
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def crop_norm(img, box):
    """Wycina obszar wg współrzędnych znormalizowanych [x, y, w, h] (0..1)."""
    H, W = img.shape[:2]
    x, y, w, h = box
    x1, y1 = int(x * W), int(y * H)
    x2, y2 = int((x + w) * W), int((y + h) * H)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return img[y1:y2, x1:x2]


def enhance_crop(bgr):
    """Uwydatnia i powiększa wycinek pola -> base64 PNG dla modelu."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape
    target_h = 200
    if h < target_h and h > 0:
        scale = min(4.0, target_h / h)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    gray = cv2.copyMakeBorder(gray, 12, 12, 12, 12,
                              cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", gray)
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --- Rozpoznawanie pojedynczego pola (serwer) ----------------------------

TYPE_HINT = {
    "name": "To pole zawiera imię i nazwisko (zwykle wielkimi literami).",
    "date": "To pole zawiera datę (dzień, miesiąc, rok).",
    "place": "To pole zawiera nazwę miejscowości lub urzędu.",
    "amount": "To pole zawiera liczbę/kwotę (mogą być zł, gr, waluta).",
    "enum": "To pole zawiera krótki wpis urzędowy (np. WŁASNOŚĆ, NIE DOTYCZY).",
    "text": "",
}

# HYBRYDA: krótkie pola czyta model wizyjny (qwen3-vl), a duże bloki
# wielolinijkowe (pojazdy, kredyty, dochody) - deepseek-ocr, bo qwen3-vl
# "myśli" na blokach bez końca i nie zwraca treści, a deepseek czyta blok
# szybko promptem "Free OCR." (na małych polach deepseek zawodzi - stąd podział).
BLOCK_MODEL = "deepseek-ocr:3b"
OCR_NATIVE_PROMPT = "Free OCR."
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*?\|>")


def is_ocr_specialist(model):
    return "ocr" in model.lower()


def clean_field_text(text):
    """Usuwa tokeny szablonowe i formatowanie Markdown/HTML z odczytu."""
    t = _SPECIAL_TOKEN_RE.sub("", text or "")
    t = re.sub(r"<[^>]+>", "", t)                         # tagi HTML
    t = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", t)       # \text{X}->X
    t = t.replace("$", "").replace("`", "")
    t = re.sub(r"[#*_]+", "", t)                          # markdown
    t = t.strip()
    if t.lower().startswith("assistant"):
        t = t[len("assistant"):].strip()
    return re.sub(r"\s+", " ", t).strip()


def recognize_field(url, model, b64, ftype):
    if is_ocr_specialist(model):
        prompt = OCR_NATIVE_PROMPT
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                   "stream": False,
                   "options": {"temperature": 0, "num_ctx": FIELD_NUM_CTX}}
    else:
        prompt = (
            "To jest powiększony wycinek JEDNEGO pola formularza. "
            "Odczytaj wyłącznie to, co zostało w tym polu napisane (ręcznie lub "
            "drukiem). Zwróć samą wartość - bez komentarza, bez etykiety pola. "
            + TYPE_HINT.get(ftype, "")
            + " Jeśli pole jest puste, napisz dokładnie: PUSTE. "
            "Jeśli zupełnie nieczytelne: [nieczytelne]."
        )
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                   "stream": False, "think": False,
                   "options": {"temperature": 0, "num_ctx": FIELD_NUM_CTX}}
    req = urllib.request.Request(
        f"{url}/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return clean_field_text(data.get("message", {}).get("content") or "")


# --- Walidacja i pewność per typ pola ------------------------------------

# --- Korekta słownikowa (RAG: duża lista polskich słów) ------------------

SLOWNIK_FILE = BASE_DIR / "slownik_polski.txt"
# Mapa "bez ogonków": OCR często gubi diakrytyki (pieniezne -> pieniężne).
_FOLD = str.maketrans("ĄĆĘŁŃÓŚŹŻ", "ACELNOSZZ")
_DICT_SET = None            # realne słowa (z ogonkami), do sprawdzenia "już OK"
_DICT_FOLDED = None         # fold -> realne słowo (przywrócenie ogonków)
_FOLDED_BY_LEN = None       # długość -> lista foldów (do ostrożnego fuzzy)


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
    if up in _DICT_SET:                       # już poprawne
        return tok
    fo = _fold(up)
    # 1) przywrócenie ogonków - dokładne i bezpieczne (SRODKI -> ŚRODKI)
    if fo in _DICT_FOLDED:
        return _dopasuj_wielkosc(tok, _DICT_FOLDED[fo])
    # 2) ostrożny fuzzy na foldach (literówki, brak/nadmiar litery)
    if not HAS_RAPIDFUZZ:
        return tok
    cands = []
    for L in (len(fo) - 1, len(fo), len(fo) + 1):
        cands += _FOLDED_BY_LEN.get(L, [])
    if not cands:
        return tok
    hit = process.extractOne(fo, cands, scorer=fuzz.ratio)
    # bardzo bliskie ORAZ ta sama długość (tylko podmiana liter, nie wstawki)
    if hit and hit[1] >= 92 and len(hit[0]) == len(fo):
        return _dopasuj_wielkosc(tok, _DICT_FOLDED[hit[0]])
    return tok


def slownik_popraw(text):
    """Poprawia polskie słowa (len>=4) w tekście wg dużego słownika:
    najpierw przywraca ogonki (bezpiecznie), potem ostrożny fuzzy.
    Nie rusza liczb, skrótów ani krótkich tokenów."""
    _load_dict()
    if not _DICT_SET:
        return text
    return re.sub(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}",
                  lambda m: _popraw_slowo(m.group(0)), text)


def _fuzzy_enum(value):
    """Ostrożne dopasowanie do słownika. NIE zawyża pewności - błędny odczyt
    nie może dostać 99% tylko dlatego, że przypadkiem pasuje do hasła."""
    if not HAS_RAPIDFUZZ:
        return value, 60
    hit = process.extractOne(value.upper(), POLSKI_SLOWNIK, scorer=fuzz.WRatio)
    if not hit:
        return value, 50
    term, score = hit[0], int(hit[1])
    if score >= 90:                 # bardzo bliskie -> podmień, ale nie 99%
        return term, min(88, score)
    if score >= 78:                 # podobne -> zaproponuj, oznacz do sprawdzenia
        return f"{term}?  (odczyt: {value})", 60
    return value, 45                # brak pewnego dopasowania -> surowy odczyt


def validate(ftype, value):
    """Zwraca (wartość_po_korekcie, pewność 0-100)."""
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
        digits = re.sub(r"\D", "", v)
        return v, (90 if len(digits) >= 2 else 45)
    if ftype == "enum":
        return _fuzzy_enum(v)
    if ftype == "name":
        return v, (85 if len(v) >= 4 and " " in v else 65)
    return v, (80 if len(v) >= 3 else 50)


# --- Główny przebieg ------------------------------------------------------

def load_template():
    return json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))


def extract(pdf_path, out_json=None, model_override=None):
    url, model = load_runtime()
    if model_override:
        model = model_override
    template = load_template()
    print(f"Serwer: {url} • model: {model}")
    results = []
    for page in template["pages"]:
        pi = page["page_index"]
        print(f"\n=== Strona {pi + 1} ===")
        img = deskew(render_page(pdf_path, pi))
        for f in page["fields"]:
            crop = crop_norm(img, f["box"])
            if crop.size == 0:
                continue
            b64 = enhance_crop(crop)
            ftype = f.get("type", "text")
            # Wszystkie pola czyta wybrany model wizyjny (qwen3-vl).
            try:
                raw = recognize_field(url, model, b64, ftype)
            except (urllib.error.URLError, OSError) as e:
                raw = f"[BŁĄD: {e}]"
            value, conf = validate(ftype, raw)
            if ftype in ("text", "place"):     # korekta słownikowa wolnego tekstu
                value = slownik_popraw(value)
            flag = "" if conf >= 70 else "  <-- SPRAWDZ"
            print(f"  {f['label']:<34} : {value!r}  ({conf}%){flag}")
            results.append({
                "page": pi + 1, "id": f["id"], "label": f["label"],
                "type": f.get("type", "text"), "value": value,
                "raw": raw, "confidence": conf,   # raw = surowy odczyt do audytu
            })
        # Zapis po każdej stronie - długi przebieg nie przepadnie.
        if out_json:
            out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    return results


def calibrate(pdf_path):
    """Rysuje ramki pól na każdej stronie -> pliki calib_strX.png do podglądu."""
    template = load_template()
    out_files = []
    for page in template["pages"]:
        pi = page["page_index"]
        img = render_page(pdf_path, pi)
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
        out_files.append(out)
        print("Zapisano:", out)
    return out_files


def main():
    # Wymuś UTF-8 na wyjściu (polska konsola Windows to domyślnie cp1250).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if len(sys.argv) < 3:
        print(__doc__)
        return
    cmd, pdf = sys.argv[1], sys.argv[2]
    if cmd == "kalibracja":
        calibrate(pdf)
    elif cmd == "rozpoznaj":
        model_override = sys.argv[3] if len(sys.argv) > 3 else None
        out_json = BASE_DIR / (Path(pdf).stem + "_pola.json")
        results = extract(pdf, out_json=out_json, model_override=model_override)
        # czytelny raport
        lines = []
        cur = None
        for r in results:
            if r["page"] != cur:
                cur = r["page"]
                lines.append(f"\n===== STRONA {cur} =====")
            mark = "" if r["confidence"] >= 70 else "   [SPRAWDŹ]"
            lines.append(f"{r['label']}: {r['value']}{mark}")
        out_txt = BASE_DIR / (Path(pdf).stem + "_pola.txt")
        out_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nZapisano: {out_json.name} oraz {out_txt.name}")
    else:
        print("Nieznana komenda. Użyj: kalibracja | rozpoznaj")


if __name__ == "__main__":
    main()
