# Zaawansowany System RAG z A/B Testing i Auto-Reindexingiem

Ten projekt to kompletny system Retrieval-Augmented Generation (RAG), który został zaprojektowany z myślą o modułowości, testowaniu i ciągłym doskonaleniu. System potrafi odpowiadać na pytania w oparciu o dostarczoną bazę wiedzy, a także dynamicznie optymalizować swoje działanie.

## 🎯 Główne Funkcjonalności

- **A/B Testing Konfiguracji**: Automatyczne testowanie czterech różnych wariantów konfiguracji w celu znalezienia najoptymalniejszych parametrów.
- **Automatyczne Re-indexowanie**: System analizuje feedback od użytkowników i w razie potrzeby samoczynnie re-indeksuje bazę wiedzy z nowymi, lepszymi parametrami.
- **Rozbudowany System Feedbacku**: Zbieranie ocen i komentarzy od użytkowników w celu monitorowania jakości odpowiedzi.
- **Dashboard Analityczny**: Możliwość wygenerowania w konsoli dashboardu z kluczowymi metrykami systemu.
- **Automatyzacja Uruchomienia**: Skrypt `run.sh` do łatwej instalacji zależności i uruchomienia aplikacji.

## ⚙️ Wymagania

- **Python 3.8+**
- Zależności wymienione w pliku `requirements.txt`.

## 🛠️ Konfiguracja

Przed pierwszym uruchomieniem należy skonfigurować dwie rzeczy:

### 1. Klucz OpenAI API

System wymaga klucza API od OpenAI do działania.

1.  Utwórz plik `.env` w głównym katalogu projektu.
2.  W pliku `.env` dodaj następującą linię, wstawiając swój klucz API:
    ```
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

### 2. Baza Wiedzy

System musi mieć dostęp do plików tekstowych, na podstawie których będzie budował swoją bazę wiedzy.

1.  Utwórz katalog `knowledge_base` w głównym katalogu projektu.
2.  Umieść w nim dowolną liczbę plików `.txt`. System automatycznie je przetworzy.

## 🚀 Jak Uruchomić

Aplikację można łatwo uruchomić za pomocą dostarczonego skryptu `run.sh`. Skrypt ten automatycznie:
1.  Sprawdzi, czy Python 3 jest zainstalowany.
2.  Utworzy wirtualne środowisko (`venv`), jeśli nie istnieje.
3.  Zainstaluje wszystkie wymagane zależności z `requirements.txt`.
4.  Sprawdzi, czy klucz API i baza wiedzy są poprawnie skonfigurowane.
5.  Uruchomi aplikację.

Aby uruchomić skrypt, wykonaj w terminalu następujące polecenia:

```bash
# Nadaj uprawnienia do wykonania skryptu (tylko za pierwszym razem)
chmod +x run.sh

# Uruchom aplikację
./run.sh
```

Po uruchomieniu aplikacja wyświetli menu z dostępnymi trybami demo.

## 🎮 Tryby Demo

1.  **Interaktywny z A/B Testing**: Testuj różne warianty konfiguracji na żywo, zadając własne pytania.
2.  **Symulacja A/B Testing**: Zobacz symulację 20 automatycznych sesji z oceną, aby porównać wydajność wariantów.
3.  **Demo Auto-reindexing**: Zobacz demonstrację, jak system automatycznie wykrywa problemy i ulepsza swoją konfigurację.
4.  **Dashboard Analityczny**: Wyświetl w konsoli pełną analitykę działania systemu.