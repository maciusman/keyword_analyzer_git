"""
Moduł odpowiedzialny za wizualizację wyników analizy słów kluczowych.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
import os

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordVisualizer:
    """
    Klasa odpowiedzialna za wizualizację wyników analizy słów kluczowych.
    """
    
    def __init__(self, output_dir: str = "data/output"):
        """
        Inicjalizuje wizualizator.
        
        Args:
            output_dir: Katalog docelowy dla plików wizualizacji
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Zainicjalizowano KeywordVisualizer z katalogiem wyjściowym: {output_dir}")
    
    def plot_cluster_summary(self, cluster_analyses: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, Any]:
        """
        Tworzy podsumowanie klastrów w postaci wykresów.
        
        Args:
            cluster_analyses: Lista słowników zawierających wyniki analizy klastrów
            top_n: Liczba najważniejszych klastrów do uwzględnienia w wykresach
            
        Returns:
            Słownik zawierający obiekty wykresów Plotly
        """
        logger.info(f"Tworzenie podsumowania klastrów (top {top_n})...")
        
        # Konwersja do DataFrame
        df = pd.DataFrame(cluster_analyses)
        
        # Ogranicz do top N klastrów według priorytetu
        top_clusters = df.sort_values(by='priority_score', ascending=False).head(top_n)
        
        # Wykres 1: Wolumen wyszukiwań według klastrów
        fig_volume = px.bar(
            top_clusters,
            x='cluster_name',
            y='total_volume',
            title='Wolumen wyszukiwań według klastrów słów kluczowych',
            labels={'cluster_name': 'Klaster', 'total_volume': 'Całkowity wolumen wyszukiwań'},
            color='priority_level',
            color_discrete_map={'Wysoki': '#ff7043', 'Średni': '#ffa726', 'Niski': '#66bb6a'},
            template='plotly_white'
        )
        fig_volume.update_layout(xaxis_tickangle=-45)
        
        # Wykres 2: Liczba słów kluczowych według klastrów
        fig_count = px.bar(
            top_clusters,
            x='cluster_name',
            y='keywords_count',
            title='Liczba słów kluczowych według klastrów',
            labels={'cluster_name': 'Klaster', 'keywords_count': 'Liczba słów kluczowych'},
            color='priority_level',
            color_discrete_map={'Wysoki': '#ff7043', 'Średni': '#ffa726', 'Niski': '#66bb6a'},
            template='plotly_white'
        )
        fig_count.update_layout(xaxis_tickangle=-45)
        
        # Wykres 3: Trudność vs. wolumen vs. liczba słów kluczowych
        fig_scatter = px.scatter(
            top_clusters,
            x='avg_difficulty',
            y='total_volume',
            size='keywords_count',
            color='priority_level',
            hover_name='cluster_name',
            title='Trudność vs. wolumen vs. liczba słów kluczowych',
            labels={
                'avg_difficulty': 'Średnia trudność (KD)',
                'total_volume': 'Całkowity wolumen wyszukiwań',
                'keywords_count': 'Liczba słów kluczowych'
            },
            color_discrete_map={'Wysoki': '#ff7043', 'Średni': '#ffa726', 'Niski': '#66bb6a'},
            template='plotly_white'
        )
        
        # Wykres 4: Rozkład priorytetów
        priority_counts = df['priority_level'].value_counts().reset_index()
        priority_counts.columns = ['priority_level', 'count']
        
        fig_priority = px.pie(
            priority_counts,
            values='count',
            names='priority_level',
            title='Rozkład priorytetów klastrów',
            color='priority_level',
            color_discrete_map={'Wysoki': '#ff7043', 'Średni': '#ffa726', 'Niski': '#66bb6a'},
            template='plotly_white'
        )
        
        # Zapisz wykresy jako HTML
        output_prefix = os.path.join(self.output_dir, "cluster")
        fig_volume.write_html(f"{output_prefix}_volume.html")
        fig_count.write_html(f"{output_prefix}_count.html")
        fig_scatter.write_html(f"{output_prefix}_scatter.html")
        fig_priority.write_html(f"{output_prefix}_priority.html")
        
        # Zwróć obiekty wykresów
        return {
            'volume': fig_volume,
            'count': fig_count,
            'scatter': fig_scatter,
            'priority': fig_priority
        }
    
    def plot_intent_distribution(self, stats: Dict[str, Any]) -> go.Figure:
        """
        Tworzy wykres rozkładu intencji.
        
        Args:
            stats: Słownik ze statystykami zawierający klucz 'intent_distribution'
            
        Returns:
            Obiekt wykresu Plotly
        """
        logger.info("Tworzenie wykresu rozkładu intencji...")
        
        intent_distribution = stats.get('intent_distribution', {})
        
        if not intent_distribution:
            logger.warning("Brak danych o intencjach. Tworzę pusty wykres.")
            fig = go.Figure()
            fig.update_layout(
                title='Brak danych o intencjach wyszukiwania',
                template='plotly_white'
            )
            return fig
        
        # Przygotuj dane do wykresu
        intents = []
        counts = []
        percentages = []
        
        for intent, data in intent_distribution.items():
            if isinstance(data, dict) and 'count' in data and 'percentage' in data:
                # Nowy format (z plików data_loader.py)
                intents.append(intent)
                counts.append(data['count'])
                percentages.append(data['percentage'])
            elif isinstance(data, (int, float)):
                # Stary format (wartość liczbowa)
                intents.append(intent)
                counts.append(int(data))
                percentages.append(round((data / sum(intent_distribution.values())) * 100, 2))
        
        # Mapowanie kolorów dla różnych intencji
        color_map = {
            'informational': '#4caf50',  # zielony
            'commercial': '#ff9800',     # pomarańczowy
            'transactional': '#e91e63',  # różowy
            'navigational': '#2196f3',   # niebieski
            'branded': '#673ab7',        # fioletowy
            'local': '#00bcd4',          # błękitny
            'unknown': '#9e9e9e'         # szary
        }
        
        # Mapowanie etykiet dla różnych intencji
        label_map = {
            'informational': 'Informacyjna',
            'commercial': 'Komercyjna',
            'transactional': 'Transakcyjna',
            'navigational': 'Nawigacyjna',
            'branded': 'Brandowa',
            'local': 'Lokalna',
            'unknown': 'Nieznana'
        }
        
        # Przygotuj DataFrame dla Plotly
        intent_df = pd.DataFrame({
            'intent': intents,
            'count': counts,
            'percentage': percentages,
            'intent_pl': [label_map.get(intent.lower(), intent) for intent in intents]
        })
        
        # Stwórz wykres
        fig = px.pie(
            intent_df,
            values='percentage',
            names='intent_pl',
            title='Rozkład intencji wyszukiwania',
            color='intent',
            color_discrete_map=color_map,
            template='plotly_white'
        )
        
        # Dodaj informację o mieszanych intencjach
        if 'mixed_intent_keywords' in stats:
            mixed_info = stats['mixed_intent_keywords']
            annotation_text = f"Uwaga: {mixed_info['percentage']}% słów kluczowych ma więcej niż jedną intencję"
            
            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=12)
            )
        
        # Zapisz wykres jako HTML
        output_path = os.path.join(self.output_dir, "intent_distribution.html")
        fig.write_html(output_path)
        
        return fig
    
    def plot_cluster_map(self, df: pd.DataFrame, cluster_names: Dict[int, str], reduced_embeddings: np.ndarray) -> go.Figure:
        """
        Tworzy mapę klastrów w przestrzeni 2D.
        
        Args:
            df: DataFrame zawierający słowa kluczowe i etykiety klastrów
            cluster_names: Słownik mapujący ID klastrów na ich nazwy
            reduced_embeddings: Tablica numpy zawierająca zredukowane embedingi
            
        Returns:
            Obiekt wykresu Plotly
        """
        if reduced_embeddings.shape[1] < 2:
            logger.error("Potrzebne są co najmniej 2 wymiary dla wizualizacji klastrów")
            return None
        
        logger.info("Tworzenie mapy klastrów...")
        
        # Stwórz DataFrame do wizualizacji
        viz_df = df[['keyword', 'cluster']].copy()
        
        # Dodaj współrzędne zredukowanych embedingów
        viz_df['x'] = reduced_embeddings[:, 0]
        viz_df['y'] = reduced_embeddings[:, 1]
        
        # Dodaj nazwy klastrów
        viz_df['cluster_name'] = viz_df['cluster'].map(lambda c: cluster_names.get(c, f"Cluster {c}") if c != -1 else "Outliers")
        
        # Mapowanie kolorów (różne dla każdego klastra)
        # Użyj -1 jako indeksu dla outlierów
        unique_clusters = sorted([c for c in df['cluster'].unique()])
        
        # Stwórz wykres
        fig = px.scatter(
            viz_df,
            x='x',
            y='y',
            color='cluster_name',
            hover_name='keyword',
            title='Mapa klastrów słów kluczowych',
            labels={'x': 'Wymiar 1', 'y': 'Wymiar 2'},
            template='plotly_white'
        )
        
        # Dostosuj wykres
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            legend_title_text='Klaster',
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02  # Poza obszarem wykresu, po prawej
            ),
            uirevision='true',
            dragmode=False
        )
        
        # Zapisz wykres jako HTML
        output_path = os.path.join(self.output_dir, "cluster_map.html")
        fig.write_html(output_path)
        
        return fig