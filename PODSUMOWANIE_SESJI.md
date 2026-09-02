# Podsumowanie sesji — od czatu z Ollamą do systemu OCR pism ręcznych

Cel, który ewoluował w trakcie: od prostego czatu z modelem, przez przepisywanie
skanów 1:1, aż po **system odczytu formularzy pole-po-polu** (oświadczenia
majątkowe) z licznikiem pewności i słownikiem RAG.

---

## 1. Co powstało — pliki

| Plik | Co to jest |
|---|---|
| `ollama_chat.py` | Prosty czat z GUI (tkinter) do serwera Ollama, historia rozmów, streaming |
| `anythingllm_chat.py` | Czat do AnythingLLM przez API, z **załącznikami** (obrazy/PDF) do przepisywania |
| `ollama_ocr_chat.py` | Aplikacja OCR: przepisywanie skanów 1:1, wybór modelu, temperatura, licznik pewności, ponawianie |
| **`field_ocr.py`** | **Główne osiągnięcie** — odczyt formularza o stałym układzie POLE-PO-POLU |
| `fields_template.json` | Szablon 5 stron oświadczenia majątkowego (~36 pól, ręcznie skalibrowane ramki) |
| `pobierz_slownik.py` | Pobiera dużą listę polskich słów (RAG do korekty) |
| `slownik_polski.txt` | 4 053 523 polskich słów (wszystkie formy odmienione, 60 MB) |
| `calib_str1..5.png` | Nakładki kalibracyjne — ramki pól narysowane na skanie |

---

## 2. Infrastruktura (ustalona w trakcie)

- **Ollama `192.168.100.53`** — modele vision: `qwen3-vl:8b` (najlepszy, ale „myśli"),
  `deepseek-ocr:3b`, `maternion/LightOnOCR-2`, `llama3.2-vision` (nie ładuje się —
  `mllama`), `llama3`.
- **Ollama `192.168.100.52`** — tylko `llama3` (tekstowy).
- **Ollama `192.168.100.12`** — `gpt-oss:20b`, `qwen3:8b`, `glm-5.3:cloud` (tekstowe).
- **AnythingLLM `192.168.100.19:3001`** — API działa (klucz **Developer API**, nie
  `brx-` który jest kluczem wtyczki przeglądarkowej!), ale jego backend Ollama był
  nieosiągalny.
- Sprzęt użytkownika: **bez GPU**, i5 11 gen, 24 GB RAM → cała obróbka obrazu lokalnie
  (CPU, tanie), rozpoznawanie na serwerze przez API.

---

## 3. Kluczowe odkrycia (droga do rozwiązania)

1. **Klucz API AnythingLLM** — `brx-...` to klucz *wtyczki przeglądarkowej*, nie działa
   z `/api/v1/`. Potrzebny osobny **Developer API Key**.

2. **Okno kontekstu (`num_ctx`)** — qwen3-vl ładuje się domyślnie z 4096 tokenami,
   a jedna strona skanu to ~2600–3800 tokenów → wiele stron dawało `HTTP 400`.
   Rozwiązanie: automatyczne dobieranie `num_ctx`.

3. **qwen3-vl „myśli" bez wyłącznika** — ignoruje `think:false` i `/no_think`.
   To „myślenie" zużywa czas i okno kontekstu; na dużych blokach model **nigdy nie
   dochodzi do treści** (zwraca puste / `[nieczytelne]`).

4. **Prędkość vs jakość modeli OCR** (przetestowane na żywo):
   - `qwen3-vl` — najlepsza jakość, ale ~20 s/pole (myślenie).
   - `deepseek-ocr` — szybki, ale na małych wycinkach zwraca pusto/500; działa tylko
     na całych stronach / dużych blokach (`Free OCR.`).
   - `LightOnOCR-2` — najszybszy (0,7 s), ale **zapętla się** („wodniwodni…"),
     wkleja LaTeX i wciąga drukowane etykiety.
   - `llama3.2-vision` — **nie uruchamia się** (stara Ollama, `unknown architecture: mllama`).

5. **Brak logprobs** — serwer (stary build) nie zwraca prawdopodobieństw tokenów,
   więc pewność liczona jest **heurystycznie** (puste/śmieciowe wyjście, artefakty
   tabel, udział sensownych znaków, liczba `[nieczytelne]`).

---

## 4. System pole-po-polu (`field_ocr.py`) — jak działa

Zamiast wysyłać całą stronę (co dawało błędy), formularz o **stałym układzie** jest
cięty na pojedyncze pola wg szablonu — mały wycinek = dużo wyższa dokładność.

```
PDF → render strony (300 DPI) → deskew (prostowanie)
    → dla KAŻDEGO pola: wytnij wycinek wg ramki [x,y,w,h]
    → uwydatnij + powiększ (OpenCV CLAHE, wyostrzenie)   ← lokalnie, CPU
    → wyślij wycinek do modelu na serwerze (API)
    → walidacja per typ (data/kwota/enum/nazwa) + pewność %
    → korekta słownikowa wolnego tekstu (RAG)
    → JSON (z surowym odczytem 'raw' do audytu) + czytelny TXT
```

**Komendy:**
```bash
python field_ocr.py kalibracja bryk.pdf   # nakładki z ramkami do podglądu
python field_ocr.py rozpoznaj  bryk.pdf   # ekstrakcja -> bryk_pola.json + .txt
```

**Funkcje:**
- Licznik **pewności %** na każde pole, `[SPRAWDZ]` przy < 70 %.
- **Zapis po każdej stronie** — długi przebieg nie przepada.
- Możliwość podania modelu z linii poleceń.
- Narzędzie **kalibracji** — ramki rysowane na skanie do dostrojenia współrzędnych.

---

## 5. Słownik RAG (`pobierz_slownik.py` + korekta w `field_ocr.py`)

- Pobrano **4 mln polskich słów** (wszystkie formy odmienione).
- Korekta oparta na **dopasowaniu „bez ogonków"** (fold): OCR gubi diakrytyki,
  więc `SRODKI→ŚRODKI`, `dochod→dochód`, `pieniezne→pieniężne`, `wlasnosc→własność`
  — dokładnie i **bezpiecznie**.
- Fuzzy tylko przy podobieństwie ≥ 92 % i tej samej długości → nie zmyśla.
- Świadomie **nie rusza** mocno zmielonego OCR, nazwisk i liczb.

---

## 6. Wyniki na `bryk.pdf` (oświadczenie majątkowe) — uczciwie

**Działa dobrze (krótkie, konkretne pola):**
- Daty (`20.03.1960`), liczby/powierzchnie (`138`, `15800`), kwoty (`13983,92`),
  pola słownikowe (`NIE DOTYCZY`, `WŁASNOŚĆ`).

**Słabo / oflagowane:**
- Nazwiska i kursywa (`JÓZEF BRYK` → `TODEF BRXIC`) — dostają niską pewność.
- **Duże bloki wielolinijkowe** (pojazdy, kredyty, dochody) — na qwen3-vl wracają
  `[nieczytelne]` (myślenie bez końca); deepseek czyta je częściowo, ale bywa
  niestabilny (pętle, 500).

**Wniosek:** podejście pole-po-polu to **duży skok** względem całej strony, a licznik
pewności skutecznie wskazuje, co sprawdzić ręcznie. Ale **wąskim gardłem pozostaje
jakość ODCZYTU pisma ręcznego przez model** — słownik naprawia ogonki, nie odczyt.

---

## 7. Rekomendacje na przyszłość (największy zysk jakości)

1. **Model trenowany na piśmie ręcznym** — `TrOCR`/`PyLaia` dotrenowany na polskich
   formularzach albo `Transkribus` z modelem PL. To jedyna droga do wysokiej
   dokładności na kursywie.
2. **Nowsza Ollama na serwerze** — odblokuje `llama3.2-vision` i **logprobs**
   (prawdziwa pewność zamiast heurystyki).
3. **Hybryda modeli** — krótkie pola do jednego modelu, bloki do innego.
4. **Gazeter nazw** (miejscowości, urzędy) do korekty nazw własnych.
