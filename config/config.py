"""
Plik konfiguracyjny zawierający ustawienia API i parametry analizy.
"""
import os
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych z pliku .env
load_dotenv()

# Klucze API (bezpieczniej jest przechowywać je w zmiennych środowiskowych)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

# Konfiguracja modeli
GEMINI_MODEL = "gemini-2.0-flash"  # Nazwa modelu Gemini
JINA_EMBEDDING_MODEL = "jina-embeddings-v2-base-en"  # Model JINA AI do embedingów

# Parametry analizy
CLUSTERING_METHOD = "hdbscan"  # Metoda klastrowania (hdbscan, dbscan, kmeans)
UMAP_N_NEIGHBORS = 15  # Parametr n_neighbors dla UMAP
UMAP_N_COMPONENTS = 5  # Liczba wymiarów po redukcji UMAP
UMAP_MIN_DIST = 0.1  # Parametr min_dist dla UMAP
HDBSCAN_MIN_CLUSTER_SIZE = 5  # Minimalny rozmiar klastra dla HDBSCAN
HDBSCAN_MIN_SAMPLES = 3  # Minimalny rozmiar próbki dla HDBSCAN

# Parametry priorytetyzacji
VOLUME_WEIGHT = 0.4      # Waga wolumenu wyszukiwań
KD_WEIGHT = 0.3          # Waga trudności słowa kluczowego
CPC_WEIGHT = 0.3         # Waga kosztu kliknięcia

# Ścieżki do plików
DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# Upewnij się, że katalogi istnieją
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
