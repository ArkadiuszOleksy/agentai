@echo off
REM ============================================================
REM  Odczyt oswiadczenia majatkowego (lokalnie, RTX 4060).
REM  Uzycie: przeciagnij plik PDF na ten plik .bat,
REM          albo uruchom dwuklikiem i podaj sciezke do PDF.
REM ============================================================
chcp 65001 >nul
setlocal
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
cd /d "%~dp0"

if "%~1"=="" (
    set /p "PDF=Przeciagnij PDF tutaj albo wpisz sciezke i Enter: "
) else (
    set "PDF=%~1"
)

"%PY%" czytaj_oswiadczenie.py "%PDF%"

echo.
echo === GOTOWE. Wyniki obok pliku PDF: *_transkrypcja.txt / .json ===
pause
