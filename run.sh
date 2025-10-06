#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Helper Functions ---
echo_green() {
    echo -e "\033[0;32m$1\033[0m"
}

echo_red() {
    echo -e "\033[0;31m$1\033[0m"
}

echo_yellow() {
    echo -e "\033[1;33m$1\033[0m"
}

# 1. --- Check for Python 3 ---
if ! command -v python3 &> /dev/null; then
    echo_red "BŁĄD: Python 3 nie jest zainstalowany. Proszę zainstalować Python 3, aby kontynuować."
    exit 1
fi

# 2. --- Check/Create Virtual Environment ---
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo_yellow "Tworzenie wirtualnego środowiska w '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# 3. --- Activate Virtual Environment ---
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo_green "Wirtualne środowisko aktywowane."

# 4. --- Install Dependencies ---
echo_yellow "Instalowanie zależności z requirements.txt..."
pip install -r requirements.txt --quiet
echo_green "Zależności zainstalowane pomyślnie."

# 5. --- Check .env file ---
if [ ! -f ".env" ]; then
    echo_yellow "Plik .env nie znaleziony. Tworzenie przykładowego pliku .env..."
    echo "OPENAI_API_KEY=\"YOUR_API_KEY_HERE\"" > .env
fi

API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2 | tr -d '"')
if [ "$API_KEY" == "YOUR_API_KEY_HERE" ] || [ -z "$API_KEY" ]; then
    echo_red "------------------------------------------------------------------"
    echo_red "BŁĄD: Klucz OpenAI API nie jest skonfigurowany."
    echo_yellow "Proszę edytować plik '.env' i wstawić swój klucz API."
    echo_red "------------------------------------------------------------------"
    exit 1
fi

# 6. --- Check knowledge_base directory ---
if [ ! -d "knowledge_base" ] || [ -z "$(ls -A knowledge_base)" ]; then
    echo_red "------------------------------------------------------------------"
    echo_red "BŁĄD: Katalog 'knowledge_base' jest pusty lub nie istnieje."
    echo_yellow "Proszę utworzyć katalog 'knowledge_base' i dodać do niego"
    echo_yellow "pliki tekstowe (.txt) do przetworzenia."
    echo_red "------------------------------------------------------------------"
    # Create the directory if it doesn't exist to help the user
    mkdir -p knowledge_base
    exit 1
fi

# 7. --- Run the Application ---
echo_green "=================================================================="
echo_green "Wszystkie sprawdzenia zakończone pomyślnie. Uruchamianie aplikacji..."
echo_green "=================================================================="
python3 rag_final.py