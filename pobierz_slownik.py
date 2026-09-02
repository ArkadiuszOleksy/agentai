"""
Pobiera dużą listę polskich słów (RAG do korekty OCR) i zapisuje jako
slownik_polski.txt (jedno słowo w linii, wielkimi literami, bez duplikatów).

Źródła (próbowane po kolei):
  1) kkrypt0nn/wordlists  - polish.txt (~3.7 MB, lista słów)
  2) ostr00000/jezyk-polski-slowniki - odm.txt (~64 MB, wszystkie odmiany)

Użycie:
    python pobierz_slownik.py            # źródło 1 (lekkie)
    python pobierz_slownik.py pelny      # źródło 2 (duże, wszystkie formy)
"""

import sys
import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).with_name("slownik_polski.txt")

SOURCES = {
    "lekki": "https://raw.githubusercontent.com/kkrypt0nn/wordlists/main/wordlists/languages/polish.txt",
    "pelny": "https://raw.githubusercontent.com/ostr00000/jezyk-polski-slowniki/master/odm.txt",
}

WORD_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{2,}")


def main():
    which = "pelny" if len(sys.argv) > 1 and sys.argv[1] == "pelny" else "lekki"
    url = SOURCES[which]
    print(f"Pobieram słownik ({which}) z:\n  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8", "replace")
    print(f"Pobrano {len(raw)/1e6:.1f} MB, przetwarzam...")

    words = set()
    for token in WORD_RE.findall(raw):
        if len(token) >= 3:
            words.add(token.upper())

    OUT.write_text("\n".join(sorted(words)), encoding="utf-8")
    print(f"Zapisano {len(words)} unikalnych słów -> {OUT.name} "
          f"({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
