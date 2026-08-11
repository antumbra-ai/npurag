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

Naprawdę biegną dwa wyszukiwania, a ich rankingi są łączone. Jedno dopasowuje po
**znaczeniu**, na embeddingach; drugie po **brzmieniu słów**, algorytmem BM25 na indeksie
pełnotekstowym. Znacznik obok score'a mówi, które znalazło dany fragment: `[v]` — po
znaczeniu, `[l]` — po słowach, `[v+l]` — oba, a `+r` dochodzi, gdy ostatnie słowo miał
reranker. Te dwa nawzajem zakrywają swoje ślepe plamy: samo znaczenie gubi numer faktury
albo kod błędu, a same słowa gubią parafrazę.

| Opcja | Działanie |
|---|---|
| `-k N` | Ile wyników zwrócić (domyślnie 8) |
| `--path WZORZEC` | Szukaj tylko w pasujących ścieżkach, np. `--path '*.md'` |
| `--mode TRYB` | `hybrid` (domyślnie), `vector` — samo znaczenie, `lexical` — same słowa |
| `--rerank TRYB` | `auto` (domyślnie), `off`, `endpoint`, `llm` — patrz niżej |
| `--json` | Wyjście maszynowe, do skryptów |

`--mode lexical` warto znać z dwóch powodów: to tryb do sięgnięcia, gdy pamiętasz dokładny
ciąg znaków, i jedyny, który nie potrzebuje żadnego serwera — działa więc również wtedy,
gdy backend leży.

### Reranking

Wyszukiwanie musi być szybkie, bo przelicza cały indeks. Reranking bierze dwadzieścia
fragmentów, które przeszły, i ogląda każdy z osobna pod kątem twojego pytania — zwykle
poprawia to kilka pierwszych wyników.

| `--rerank` | Co się dzieje |
|---|---|
| `auto` | Rerankuje, jeśli backend ma model rerankujący; jeśli nie ma — pomija. Domyślne. |
| `off` | Ranking wyłącznie ze score'ów wyszukiwania. |
| `endpoint` | Żąda modelu rerankującego z backendu i kończy błędem, gdy go nie ma. |
| `llm` | Ocenia fragmenty modelem czatowym. Działa na każdym backendzie, kosztuje generację. |

`auto` nic nie robi, dopóki nie wpiszesz backendowi `rerank_model` w konfiguracji — npurag
nie zakłada, że masz załadowany trzeci model. `npurag status` pokazuje, czy jakiś jest.

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
| `--mode TRYB` | Jak szukać fragmentów; tryby jak w `search` |
| `--rerank TRYB` | Jak rerankować krótką listę; tryby jak w `search` |
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

### Udostępnij indeks asystentowi

```bash
npurag mcp ~/Dokumenty
```

To wystawia indeks po **Model Context Protocol**, dzięki czemu asystent sam przeszukuje
twoje pliki, zamiast czekać, aż wkleisz mu fragmenty. To nie jest usługa w tle: nic nie
nasłuchuje na porcie i nie uruchamiasz tego polecenia ręcznie. Asystent odpala npuraga jako
proces potomny i rozmawia z nim przez potok — i właśnie dlatego całość zostaje na twojej
maszynie.

Klienta podłączasz wpisem w jego konfiguracji MCP:

```json
{
  "mcpServers": {
    "npurag": {
      "command": "npurag",
      "args": ["mcp", "/home/ty/Dokumenty"]
    }
  }
}
```

Podaj pełną ścieżkę do binarki, jeśli nie leży na `PATH`, który widzi klient, i wskaż
katalog, który zaindeksowałeś — inaczej niż `search`, ta komenda nie zgaduje go z katalogu
bieżącego, bo katalog bieżący wybiera klient, a nie ty.

Wystawione są trzy narzędzia:

| Narzędzie | Działanie |
|---|---|
| `search` | Zwraca same pasujące fragmenty wraz z plikiem, z którego pochodzą |
| `ask` | Każe lokalnemu modelowi napisać odpowiedź z tych fragmentów, ze źródłami |
| `status` | Mówi, co obejmuje indeks i czy backend odpowiada |

`search` i `ask` przyjmują te same pokrętła co komendy: `k`, `path`, `mode` i `rerank`.
Asystent, który potrzebuje dokładnego ciągu znaków, sam poprosi o `mode: "lexical"`.

### Udostępnij indeks programowi

```bash
npurag serve ~/Dokumenty
```

Dla wołających, którzy nie są asystentem — skryptu, usługi, zadania z crona. Odpowiada pod
`http://127.0.0.1:8787` tym samym JSON-em, który drukuje `--json`:

```bash
curl 'http://127.0.0.1:8787/search?q=retencja%20backupów&k=5'
curl -H 'Content-Type: application/json' \
     -d '{"question": "co ustaliłem w sprawie projektu X?"}' \
     http://127.0.0.1:8787/ask
```

| Trasa | Działanie |
|---|---|
| `GET`/`POST` `/search` | Pasujące fragmenty, tak jak zwraca je `search --json` |
| `GET`/`POST` `/ask` | Odpowiedź ze źródłami, tak jak zwraca ją `ask --json` |
| `GET` `/status` | Co obejmuje indeks i czy backend odpowiada |
| `GET` `/health` | Żyje czy nie — bez żadnych poświadczeń |

Argumenty idą w query stringu albo w ciele JSON: `q` (lub `query`/`question`), `k`, `path`,
`mode`, `rerank`, a dla `/ask` również `model`. Gdy podasz oba, wygrywa ciało.

**Przez ten port czyta się cały indeks.** Dlatego domyślnie nasłuchuje wyłącznie na
loopbacku i **odmówi** nasłuchiwania gdziekolwiek indziej, dopóki nie ustawisz tokenu:

```bash
NPURAG_TOKEN=$(openssl rand -hex 16) npurag serve ~/Dokumenty --bind 0.0.0.0:8787
```

Wołający wysyła wtedy `Authorization: Bearer <token>`. Używaj zmiennej środowiskowej, nie
`--token`: argument widzi każdy, kto potrafi wylistować procesy. `/health` zostaje otwarte,
żeby monitoring mógł sondować bez trzymania poświadczeń.

Żądania obsługiwane są pojedynczo. To indeks osobisty, nie serwis webowy, a drugie żądanie
czekające na pierwsze jest lepszym kompromisem niż pula uchwytów do bazy.

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

[search]
mode           = "hybrid"    # hybrid | vector | lexical
rerank         = "auto"      # auto | off | endpoint | llm
rerank_top     = 20          # ile fragmentów dostaje reranker
candidates     = 0           # ilu kandydatów na wyszukiwanie przed łączeniem; 0 = z -k
rrf_k          = 60.0        # jak płasko oba rankingi ważą się nawzajem
vector_weight  = 1.0         # podnieś, żeby bardziej ufać znaczeniu
lexical_weight = 1.0         # podnieś, żeby bardziej ufać dokładnym słowom

[backends.amd-flm]
base_url    = "http://localhost:52625/v1"
embed_model = "embeddinggemma-300m"
chat_model  = "gemma3:4b"

[backends.intel-ovms]
base_url    = "http://localhost:8000/v3"
embed_model = "embeddinggemma-300m"
chat_model  = "gemma3:4b-int4-ov"
# rerank_model = "bge-reranker-base"   # jeśli twój serwer taki ma
```

Backend przełączysz na jedno wywołanie przez `--backend intel-ovms`, a sam adres przez
`--base-url`. Działają też zmienne środowiskowe: `NPURAG_BACKEND`, `NPURAG_BASE_URL`,
`NPURAG_EMBED_MODEL`, `NPURAG_CHAT_MODEL`, `NPURAG_RERANK_MODEL`, `NPURAG_DB`. Flagi z
linii poleceń wygrywają ze zmiennymi środowiskowymi, a te z plikiem konfiguracyjnym.

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
daje nieporównanie więcej do dopasowania niż pojedyncze słowo. Jeśli pamiętasz dokładne
brzmienie, `--mode lexical` poszuka go dosłownie; jeśli pamiętasz tylko sens, `--mode
vector` całkowicie zignoruje słowa. `--rerank llm` przyjrzy się krótkiej liście dokładniej,
kosztem jednej generacji. Rozmiar fragmentów zmienia `chunk_tokens`, a `--path` zawęża
wyszukiwanie, gdy z grubsza wiesz, gdzie leży odpowiedź.

**Indeks zbudowany starszą wersją.** Zostanie uaktualniony w miejscu przy pierwszym
otwarciu: połowa pełnotekstowa powstaje z tekstu, który indeks już trzyma, więc nic nie
idzie do backendu i nic nie wymaga ponownego liczenia embeddingów.

---

Licencja Apache 2.0. Zgłoszenia i pytania:
<https://github.com/antumbra-ai/npurag/issues>
