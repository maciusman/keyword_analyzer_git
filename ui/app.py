"""
Aplikacja Streamlit do analizy słów kluczowych.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

# Dodaj katalog projektu do ścieżki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importy z projektu
from utils.data_loader import load_and_prepare_data, save_analysis, load_analysis
from models.embeddings import EmbeddingGenerator
from models.clustering import KeywordClusterer
from models.analyzer import KeywordAnalyzer
from utils.visualizer import KeywordVisualizer
from config.config import DATA_DIR, OUTPUT_DIR

# Konfiguracja strony
st.set_page_config(
    page_title="Analizator Słów Kluczowych",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tytuł aplikacji
st.title("Analizator Słów Kluczowych")
st.markdown("**Narzędzie do klastrowania i analizy słów kluczowych z wykorzystaniem AI**")

# Panel boczny
st.sidebar.header("Ustawienia")

# Sekcja importu/eksportu analizy
st.sidebar.subheader("Import/Eksport analizy")

# Przycisk do zapisania analizy - widoczny zawsze
if st.sidebar.button("💾 Zapisz analizę", type="secondary"):
    # Sprawdź czy mamy dane do zapisania
    if 'clustered_df' in st.session_state:
        # Zbierz wszystkie dane z sesji do zapisania
        analysis_data = {
            'keywords_df': st.session_state.get('keywords_df'),
            'stats': st.session_state.get('stats'),
            'df_with_embeddings': st.session_state.get('df_with_embeddings'),
            'reduced_embeddings': st.session_state.get('reduced_embeddings'),
            'clustered_df': st.session_state.get('clustered_df'),
            'cluster_names': st.session_state.get('cluster_names'),
            'cluster_analyses': st.session_state.get('cluster_analyses'),
            'plots': st.session_state.get('plots'),
            'intent_plot': st.session_state.get('intent_plot'),
            'cluster_map': st.session_state.get('cluster_map')
        }
        
        # Zapisz analizę
        save_path = save_analysis(analysis_data)
        st.sidebar.success(f"Analiza zapisana: {os.path.basename(save_path)}")
    else:
        # Komunikat jeśli nie ma danych do zapisania
        st.sidebar.error("Brak danych do zapisania. Wykonaj analizę przed zapisaniem.")

# Wczytywanie zapisanej analizy
import_file = st.sidebar.file_uploader("Wczytaj zapisaną analizę (.pkl)", type=["pkl"])
if import_file:
    try:
        # Zapisz plik tymczasowo
        temp_path = os.path.join(DATA_DIR, import_file.name)
        with open(temp_path, "wb") as f:
            f.write(import_file.getbuffer())
        
        # Wczytaj analizę
        loaded_data = load_analysis(temp_path)
        
        # Przywróć dane do sesji
        for key, value in loaded_data.items():
            st.session_state[key] = value
        
        st.session_state.analysis_completed = True
        st.session_state.file_imported = True
        st.session_state.imported_file_name = import_file.name
        
        st.sidebar.success(f"Analiza wczytana: {import_file.name}")
        st.rerun()  # Odśwież aplikację aby zaktualizować wyświetlane dane
        
    except Exception as e:
        st.sidebar.error(f"Błąd podczas wczytywania analizy: {e}")

# Oddzielamy sekcje w bocznym panelu
st.sidebar.markdown("---")
st.sidebar.subheader("Nowa analiza")

# Funkcja do wczytywania i przetwarzania danych
def process_keywords(file_path):
    """
    Przetwarza słowa kluczowe z pliku.
    
    Args:
        file_path: Ścieżka do pliku z danymi
        
    Returns:
        Tuple zawierający wyniki analizy i wizualizacje
    """
    # Inicjalizacja pasków postępu
    progress_col, status_col = st.columns([3, 2])
    overall_progress_bar = progress_col.progress(0)
    overall_status = progress_col.empty()
    
    step_progress_bar = progress_col.progress(0)
    step_status = progress_col.empty()
    
    current_step_container = status_col.empty()
    overall_status_container = status_col.empty()
    
    # Funkcja callback do aktualizacji pasków postępu
    def update_progress(overall_progress, step_progress, overall_text, step_text, step_name):
        overall_progress_bar.progress(overall_progress)
        overall_status.text(overall_text)
        step_progress_bar.progress(step_progress)
        step_status.text(step_text)
        current_step_container.markdown(f"### Obecny krok: {step_name}")
    
    # Krok 1: Wczytaj i przygotuj dane (10%)
    update_progress(0.0, 0.0, "Inicjalizacja...", "Wczytywanie pliku...", "Przetwarzanie danych")
    
    df, stats = load_and_prepare_data(file_path)
    st.session_state.keywords_df = df
    st.session_state.stats = stats
    
    update_progress(0.1, 1.0, "Przetwarzanie danych zakończone", "Plik wczytany", "Przetwarzanie danych")
    
    # Krok 2: Generuj embeddingi (10% - 30%)
    embedding_generator = EmbeddingGenerator()
    
    def embedding_callback(progress, status):
        update_progress(0.1 + 0.2 * progress, progress, 
                       "Generowanie embedingów...", 
                       status, 
                       "Generowanie embedingów")
    
    df_with_embeddings = embedding_generator.process_keywords_dataframe(df, progress_callback=embedding_callback)
    st.session_state.df_with_embeddings = df_with_embeddings
    
    # Krok 3: Klastruj słowa kluczowe (30% - 50%)
    clusterer = KeywordClusterer()
    
    def clustering_callback(progress, status):
        update_progress(0.3 + 0.2 * progress, progress,
                       "Klastrowanie słów kluczowych...",
                       status,
                       "Klastrowanie")
    
    clustered_df = clusterer.process_keywords_dataframe(df_with_embeddings, progress_callback=clustering_callback)
    st.session_state.reduced_embeddings = clusterer.reduced_embeddings
    st.session_state.clustered_df = clustered_df
    
    # Krok 4: Nazwij klastry (50% - 65%)
    analyzer = KeywordAnalyzer()
    
    def naming_callback(progress, status):
        update_progress(0.5 + 0.15 * progress, progress,
                       "Nazywanie klastrów...",
                       status,
                       "Nazywanie klastrów")
    
    cluster_names = analyzer.name_clusters(clustered_df, progress_callback=naming_callback)
    st.session_state.cluster_names = cluster_names
    
    # Krok 5: Analizuj klastry (65% - 85%)
    def analysis_callback(progress, status):
        update_progress(0.65 + 0.2 * progress, progress,
                       "Analizowanie klastrów...",
                       status,
                       "Analiza klastrów")
    
    cluster_analyses = analyzer.process_all_clusters(clustered_df, cluster_names, progress_callback=analysis_callback)
    st.session_state.cluster_analyses = cluster_analyses
    
    # Krok 6: Twórz wizualizacje (85% - 100%)
    update_progress(0.85, 0.0, "Tworzenie wizualizacji...", "Przygotowywanie wykresów...", "Wizualizacja")
    
    visualizer = KeywordVisualizer()
    
    # Podsumowanie klastrów
    update_progress(0.90, 0.3, "Tworzenie wizualizacji...", "Wykres podsumowania klastrów...", "Wizualizacja")
    plots = visualizer.plot_cluster_summary(cluster_analyses)
    st.session_state.plots = plots
    
    # Rozkład intencji - ZMIANA: przekazanie stats zamiast DataFrame
    update_progress(0.93, 0.6, "Tworzenie wizualizacji...", "Wykres rozkładu intencji...", "Wizualizacja")
    intent_plot = visualizer.plot_intent_distribution(stats)
    st.session_state.intent_plot = intent_plot
    
    # Mapa klastrów
    update_progress(0.96, 0.9, "Tworzenie wizualizacji...", "Mapa klastrów...", "Wizualizacja")
    cluster_map = visualizer.plot_cluster_map(clustered_df, cluster_names, clusterer.reduced_embeddings)
    st.session_state.cluster_map = cluster_map
    
    # Zakończenie
    update_progress(1.0, 1.0, "Analiza zakończona", "Wszystkie wizualizacje gotowe", "Zakończono")
    time.sleep(0.5)  # Chwilowa pauza przed wyświetleniem wyników
    
    st.success("Analiza zakończona!")
    
    # Usuń paski postępu
    progress_col.empty()
    status_col.empty()

# Wczytywanie pliku
upload_file = st.sidebar.file_uploader("Wczytaj plik CSV/Excel z Ahrefs", type=["csv", "xlsx", "xls"])

if upload_file:
    # Zapisz plik tymczasowo
    file_path = os.path.join(DATA_DIR, upload_file.name)
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
    
    # Zapisz informację o wczytanym pliku w sesji
    st.session_state.file_uploaded = True
    st.session_state.file_path = file_path
    st.session_state.file_name = upload_file.name
    
    # Wyświetl informację o wczytanym pliku
    st.sidebar.success(f"Plik wczytany: {upload_file.name}")
    
    # Przycisk do uruchomienia analizy
    if st.sidebar.button("🚀 Uruchom analizę", type="primary"):
        # Wyczyść poprzednie wyniki jeśli istnieją
        keys_to_clear = ['clustered_df', 'cluster_analyses', 'plots', 'intent_plot', 'cluster_map', 
                         'keywords_df', 'stats', 'df_with_embeddings', 'reduced_embeddings', 'cluster_names']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Uruchom analizę
        process_keywords(file_path)
        st.session_state.analysis_completed = True
    
    # Przycisk do ponownego przetworzenia (jeśli analiza była już wykonana)
    if 'analysis_completed' in st.session_state and st.session_state.analysis_completed:
        if st.sidebar.button("↻ Analizuj ponownie"):
            # Resetujemy sesję
            for key in list(st.session_state.keys()):
                if key not in ['file_uploaded', 'file_path', 'file_name']:
                    del st.session_state[key]
            process_keywords(file_path)
            st.session_state.analysis_completed = True

# Wyświetlanie wyników
if 'clustered_df' in st.session_state:
    # Podstawowe statystyki
    st.header("Przegląd danych")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Liczba słów kluczowych", st.session_state.stats['total_keywords'])
    
    with col2:
        st.metric("Łączny wolumen wyszukiwań", st.session_state.stats['total_volume'])
    
    with col3:
        st.metric("Średnia trudność (KD)", f"{st.session_state.stats['avg_difficulty']:.2f}")
    
    # Rozkład intencji
    st.header("Rozkład intencji wyszukiwania")
    st.plotly_chart(st.session_state.intent_plot, use_container_width=True)
    
    # Mapa klastrów
    st.header("Mapa klastrów słów kluczowych")
    st.plotly_chart(st.session_state.cluster_map, use_container_width=True)
    
    # Wizualizacje klastrów
    st.header("Podsumowanie klastrów")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Wolumen wyszukiwań według klastrów")
        st.plotly_chart(st.session_state.plots['volume'], use_container_width=True)
    
    with col2:
        st.subheader("Liczba słów kluczowych według klastrów")
        st.plotly_chart(st.session_state.plots['count'], use_container_width=True)
    
    st.subheader("Trudność vs. wolumen vs. liczba słów kluczowych")
    st.plotly_chart(st.session_state.plots['scatter'], use_container_width=True)
    
    st.subheader("Rozkład priorytetów klastrów")
    st.plotly_chart(st.session_state.plots['priority'], use_container_width=True)
    
    # Szczegółowe wyniki dla każdego klastra
    st.header("Szczegółowa analiza klastrów")
    
    for cluster_analysis in st.session_state.cluster_analyses:
        # Określenie koloru dla priorytetu
        if cluster_analysis['priority_level'] == 'Wysoki':
            priority_color = "#ff7043"  # pomarańczowy
        elif cluster_analysis['priority_level'] == 'Średni':
            priority_color = "#ffa726"  # żółty
        else:
            priority_color = "#66bb6a"  # zielony
        
        # Tworzenie nagłówka z kolorowym znacznikiem priorytetu
        st.markdown(
            f"""
            <h3 style="margin-bottom: 0px;">
                {cluster_analysis['cluster_name']} 
                <span style="color: {priority_color}; font-size: 0.8em;">
                    [{cluster_analysis['priority_level']} priorytet]
                </span>
            </h3>
            """, 
            unsafe_allow_html=True
        )
        
        # Metryki klastra
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Liczba słów kluczowych", cluster_analysis['keywords_count'])
        
        with col2:
            st.metric("Łączny wolumen", cluster_analysis['total_volume'])
        
        with col3:
            st.metric("Średnia trudność", f"{cluster_analysis['avg_difficulty']:.2f}")
        
        with col4:
            st.metric("Średni CPC", f"${cluster_analysis['avg_cpc']:.2f}")
        
        # Wnioski i rekomendacje
        st.markdown("**Wnioski:**")
        st.write(cluster_analysis['insights'])
        
        st.markdown("**Rekomendacje dla treści:**")
        st.write(cluster_analysis['content_strategy'])
        
        # Słowa kluczowe w klastrze
        with st.expander("Zobacz słowa kluczowe w tym klastrze"):
            cluster_id = cluster_analysis['cluster_id']
            cluster_keywords = st.session_state.clustered_df[st.session_state.clustered_df['cluster'] == cluster_id]
            cluster_keywords = cluster_keywords.sort_values(by='volume', ascending=False)
            
            # Sprawdzenie jakie kolumny intencji są dostępne
            intent_columns = ['branded', 'local', 'navigational', 'informational', 'commercial', 'transactional']
            available_intent_columns = [col for col in intent_columns if col in cluster_keywords.columns]
            display_columns = ['keyword', 'volume', 'difficulty', 'cpc']
            
            # Dodaj kolumny intencji, jeśli są dostępne
            if available_intent_columns:
                display_columns.extend(available_intent_columns)
            elif 'intent' in cluster_keywords.columns:
                display_columns.append('intent')
                
            st.dataframe(cluster_keywords[display_columns])
        
        st.markdown("---")
    
    # Opcja eksportu wyników
    st.header("Eksport wyników")
    
    # Eksport oryginalnych danych z klastrami
    if st.button("Eksportuj wszystkie słowa kluczowe z klastrami"):
        # Przygotuj dane do eksportu
        export_df = st.session_state.clustered_df.copy()
        
        # Dodaj nazwy klastrów
        export_df['cluster_name'] = export_df['cluster'].map(
            lambda c: st.session_state.cluster_names.get(c, f"Cluster {c}") if c != -1 else "Outliers"
        )
        
        # Usuń kolumny z embedingami i inne duże obiekty przed zapisem
        columns_to_drop = []
        if 'embedding' in export_df.columns:
            columns_to_drop.append('embedding')
        
        # Przekształć listę intencji na string przed zapisem
        if 'intent_list' in export_df.columns:
            export_df['intent_list'] = export_df['intent_list'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            
        if columns_to_drop:
            export_df = export_df.drop(columns=columns_to_drop)
        
        # Zapisz do pliku
        export_path = os.path.join(OUTPUT_DIR, "keywords_with_clusters.csv")
        export_df.to_csv(export_path, index=False)
        
        # Link do pobrania
        st.markdown(f"Plik został zapisany w: `{export_path}`")
    
    # Eksport analizy klastrów
    if st.button("Eksportuj analizę klastrów"):
        # Przygotuj dane do eksportu
        export_df = pd.DataFrame(st.session_state.cluster_analyses)
        
        # Zapisz do pliku
        export_path = os.path.join(OUTPUT_DIR, "cluster_analysis.csv")
        export_df.to_csv(export_path, index=False)
        
        # Link do pobrania
        st.markdown(f"Plik został zapisany w: `{export_path}`")

else:
    # Instrukcje dla użytkownika
    if 'file_imported' in st.session_state and st.session_state.file_imported:
        st.info(f"Wczytano zapisaną analizę: {st.session_state.imported_file_name}")
    else:
        st.info("Wczytaj plik CSV lub Excel z eksportu Ahrefs, aby rozpocząć analizę, lub zaimportuj zapisaną analizę.")
    
    # Przykładowa struktura danych
    st.markdown("""
    ### Oczekiwana struktura danych
    
    Plik powinien zawierać co najmniej następujące kolumny:
    - `keyword` - fraza kluczowa
    - `volume` - miesięczny wolumen wyszukiwań
    - `difficulty` / `kd` - trudność słowa kluczowego
    - `intent` - intencja wyszukiwania (informational, commercial, transactional, navigational)
    - `cpc` - koszt kliknięcia
    
    Dodatkowo, dla rozkładu intencji, aplikacja obsługuje kolumny:
    - `branded` - intencja brandowa (True/False)
    - `local` - intencja lokalna (True/False)
    - `navigational` - intencja nawigacyjna (True/False)
    - `informational` - intencja informacyjna (True/False)
    - `commercial` - intencja komercyjna (True/False)
    - `transactional` - intencja transakcyjna (True/False)
    
    Nazwy kolumn mogą nieznacznie się różnić - aplikacja spróbuje je zmapować automatycznie.
    """)