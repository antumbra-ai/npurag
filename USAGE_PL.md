# npurag — instrukcja użycia

npurag zamienia wskazany katalog na Twoim dysku w indeks, po którym można szukać
**znaczeniem**, a nie dokładnym słowem, oraz zadawać pytania, na które odpowiedź powstaje
z Twoich własnych plików — wraz z fragmentami, z których została zbudowana.

Nic nie jest nigdzie wysyłane. Indeks to pojedynczy plik SQLite na Twoim dysku, a model
czytający Twój tekst działa na Twojej maszynie.

## Zanim zaczniesz

npurag sam nie uruchamia modelu. Rozmawia po HTTP z lokalnym serwerem, więc taki serwer
musi działać:

- **AMD Ryzen AI (FastFlowLM)** — `flm serve gemma3:4b --embed 1`, nasłuchuje pod
  `http://localhost:52625/v1`. To ustawienie domyślne.
- **Intel (OpenVINO Model Server)** — nasłuchuje na porcie 8000. Uwaga: prefiks API bywa
  `/v3`, a nie `/v1` — sprawdź swoją wersję i ustaw `base_url` zgodnie z nią.
- **Cokolwiek innego, co mówi po API OpenAI** — wskaż to przez `--base-url`.

Sprawdź, co npurag widzi:

```bash
npurag status
```

Wypisze aktywny backend, adres, pod który zadzwoni, czy ten adres odpowiedział, oraz
rozmiar indeksu. Jeśli widzisz `unreachable`, serwer nie działa albo jest na innym porcie
— napraw to przed indeksowaniem.

Chcesz zobaczyć npurag bez żadnego serwera? Dodaj `--mock`. Użyje wbudowanego zamiennika,
który nie wymaga sprzętu. Wyniki są zgrubne, ale wszystkie komendy działają — to dobry
sposób, żeby poznać całość.

## Cztery komendy

### Zaindeksuj katalog

```bash
npurag index ~/Dokumenty
```

Pierwszy przebieg czyta wszystko. Kolejne czytają tylko to, co się zmieniło: plik o
niezmienionym rozmiarze i dacie nie jest w ogóle otwierany, a plik jedynie zapisany
ponownie bez zmian zostaje rozpoznany po zawartości i nie jest przeliczany. Dzięki temu
uruchamianie tego z timera jest tanie.

| Opcja | Działanie |
|---|---|
| `--reindex` | Przebuduj od zera, ignorując dotychczasowy indeks |
| `--include WZORZEC` | Indeksuj wyłącznie pasujące pliki; można powtarzać |
| `--exclude WZORZEC` | Pomiń pasujące pliki, ponad wykluczenia z konfiguracji |
| `--max-size MB` | Pomiń pliki większe niż podany rozmiar (domyślnie 5 MB) |
| `--follow-symlinks` | Podążaj za dowiązaniami zamiast je pomijać |

Podsumowanie na końcu raportuje wszystko, co pominięto i dlaczego — pliki binarne, zbyt
duże, formaty wymagające parsera, którego nie ma. Przebieg, który zaindeksował mniej niż
oczekiwałeś, powie o tym wprost, zamiast wyglądać na udany.

### Szukaj znaczeniem

```bash
npurag search "jak skonfigurowałem backup"
```

Dostajesz najlepiej pasujące fragmenty ze score'em, plik źródłowy i podgląd. Słowa nie
muszą się zgadzać: notatka o „nocnej retencji archiwów" może odpowiedzieć na pytanie o
backup.

| Opcja | Działanie |
|---|---|
| `-k N` | Ile wyników zwrócić (domyślnie 8) |
| `--path WZORZEC` | Szukaj tylko w pasujących ścieżkach, np. `--path '*.md'` |
| `--json` | Wyjście maszynowe, do skryptów |

### Zadaj pytanie

```bash
npurag ask "co ustaliłem w sprawie projektu X?"
```

npurag znajduje istotne fragmenty, podaje je modelowi i drukuje odpowiedź, a pod nią
sekcję **Sources**: z jakiego zbioru korzystał oraz z którego pliku i fragmentu pochodzi
każdy cytat. Model ma polecenie odpowiadać wyłącznie z tych fragmentów i przyznać, gdy nie
ma w nich odpowiedzi — traktuj więc pewnie brzmiącą odpowiedź bez źródeł jako ostrzeżenie.

| Opcja | Działanie |
|---|---|
| `-k N` | Z ilu fragmentów korzystać (domyślnie 8) |
| `--path WZORZEC` | Czerp tylko z pasujących ścieżek |
| `--model NAZWA` | Użyj innego modelu czatowego do tego pytania |
| `--no-sources` | Wypisz samą odpowiedź |
| `--json` | Odpowiedź, źródła i pochodzenie jako JSON |

Pytaj w dowolnym języku — model ma polecenie odpowiadać w języku pytania.

### Utrzymuj świeżość

```bash
npurag watch ~/Dokumenty   # reindeksuje przy zmianach; zatrzymanie Ctrl-C
npurag prune               # usuwa wpisy plików, których już nie ma
```

`watch` czeka, aż edycja się uspokoi, więc jeden zapis pliku powoduje jedną aktualizację,
a nie kilka. Jeśli wolisz nie trzymać uruchomionego programu, zaplanuj `npurag index` —
na Linuksie projekt dostarcza gotowe unity systemd-user właśnie do tego.

## Gdzie co leży

- **Indeks** — `~/.local/share/npurag/<id>/index.db` na Linuksie,
  `%LOCALAPPDATA%\npurag\data\<id>\index.db` na Windowsie. Jeden indeks na katalog.
  Skasowanie go nie traci nic poza czasem potrzebnym na odbudowę.
- **Konfiguracja** — `~/.config/npurag/config.toml` na Linuksie,
  `%APPDATA%\npurag\config\config.toml` na Windowsie. Jest opcjonalna; domyślne
  ustawienia działają.

Plik konfiguracyjny wygląda tak:

```toml
backend = "amd-flm"          # który preset poniżej jest aktywny

max_file_size_mb = 5
chunk_tokens     = 400       # jak duży jest mniej więcej jeden indeksowany fragment
chunk_overlap    = 60        # ile sąsiadujące fragmenty mają wspólnego
exclude = [".git/**", "node_modules/**", "target/**", "**/*.min.js"]
external_extractors = true   # wolno wołać pdftotext / pandoc, jeśli są zainstalowane

[backends.amd-flm]
base_url    = "http://localhost:52625/v1"
embed_model = "embeddinggemma-300m"
chat_model  = "gemma3:4b"

[backends.intel-ovms]
base_url    = "http://localhost:8000/v3"
embed_model = "embeddinggemma-300m"
chat_model  = "gemma3:4b-int4-ov"
```

Backend przełączysz na jedno wywołanie przez `--backend intel-ovms`, a sam adres przez
`--base-url`. Działają też zmienne środowiskowe: `NPURAG_BACKEND`, `NPURAG_BASE_URL`,
`NPURAG_EMBED_MODEL`, `NPURAG_CHAT_MODEL`, `NPURAG_DB`. Flagi z linii poleceń wygrywają ze
zmiennymi środowiskowymi, a te z plikiem konfiguracyjnym.

## Jakie typy plików

Tekst, kod źródłowy i Markdown działają zawsze. PDF, HTML oraz dokumenty biurowe (DOCX,
PPTX, XLSX, ODT, ODP) działają w gotowych paczkach do pobrania, bo zawierają te parsery.
Czego npurag nie potrafi przeczytać, to pomija i zlicza — nigdy nie gubi po cichu.

Jeśli w systemie są `pdftotext` lub `pandoc`, npurag użyje ich dla formatów, których sam
nie czyta, w tym starszych jak `.doc` czy `.ods`. To programy lokalne; ustaw
`external_extractors = false`, jeśli wolisz, żeby npurag nie uruchamiał żadnego procesu.

## Gdy coś nie gra

**`status` mówi, że backend jest nieosiągalny.** Serwer nie działa, stoi na innym porcie
albo ma inny prefiks API — OpenVINO Model Server często używa `/v3` tam, gdzie FastFlowLM
używa `/v1`. Adres pokazany przez `status` to dokładnie ten, pod który npurag zadzwoni.

**„this index was built with embedding model X".** Indeks ma sens wyłącznie razem z
modelem, który go zbudował; wektory z dwóch różnych modeli nie są porównywalne. Uruchom
`npurag index <katalog> --reindex`, żeby przebudować go aktualnie skonfigurowanym modelem.

**„no index for … yet".** `search` i `ask` korzystają z indeksu dla katalogu, w którym
aktualnie jesteś. Albo wejdź (`cd`) do zaindeksowanego katalogu, albo podaj `--db` ze
ścieżką do jego indeksu.

**Wyniki wyglądają słabo.** Spróbuj dłuższego, bardziej konkretnego pytania — pełne zdanie
daje nieporównanie więcej do dopasowania niż pojedyncze słowo. Rozmiar fragmentów zmienia
`chunk_tokens`, a `--path` zawęża wyszukiwanie, gdy z grubsza wiesz, gdzie leży odpowiedź.

---

Licencja Apache 2.0. Zgłoszenia i pytania:
<https://github.com/antumbra-ai/npurag/issues>
