"""
Moduł analizujący słowa kluczowe przy użyciu modelu Gemini.
Zoptymalizowany dla wysokiej jakości analizy z pełnym kontekstem słów kluczowych.
"""
import logging
import pandas as pd
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
from tqdm import tqdm
import time

# Import konfiguracji
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    GEMINI_API_KEY, GEMINI_MODEL, 
    VOLUME_WEIGHT, KD_WEIGHT, CPC_WEIGHT
)

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    """
    Klasa odpowiedzialna za analizę słów kluczowych z wykorzystaniem modelu Gemini.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Inicjalizuje analizator słów kluczowych.
        
        Args:
            api_key: Klucz API dla Gemini (opcjonalnie, domyślnie z config)
            model_name: Nazwa modelu Gemini (opcjonalnie, domyślnie z config)
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_MODEL
        
        # Sprawdzenie, czy klucz API jest dostępny
        if not self.api_key:
            raise ValueError("Brak klucza API dla Gemini. Ustaw go w pliku .env.")
        
        # Konfiguracja Gemini
        try:
            genai.configure(api_key=self.api_key)
            
            # Inicjalizacja z optymalnymi parametrami dla jakości
            generation_config = {
                "temperature": 0.2,  # Niższa temperatura dla bardziej deterministycznych odpowiedzi
                "top_p": 0.95,       # Wysoka wartość top_p dla większej różnorodności, ale zachowania precyzji
                "top_k": 40,         # Standardowa wartość top_k
                "max_output_tokens": 4096,  # Dłuższe odpowiedzi dla bardziej szczegółowej analizy
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Ustaw bezpieczniejszy model - jeśli flash nie działa
            try:
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Testowe wywołanie aby sprawdzić czy działa
                test_response = self.model.generate_content("Say hello")
                if not test_response.text:
                    logger.warning(f"Model {self.model_name} zwrócił pustą odpowiedź. Próbuję innego modelu.")
                    self.model_name = "gemini-1.5-pro"
                    self.model = genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
            except Exception as e:
                logger.warning(f"Problem z modelem {self.model_name}: {e}. Przełączam na gemini-1.5-pro.")
                self.model_name = "gemini-1.5-pro"
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
            logger.info(f"Zainicjalizowano model Gemini: {self.model_name}")
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji modelu Gemini: {e}")
            raise
    
    def name_clusters(self, df: pd.DataFrame, max_keywords_per_cluster: Optional[int] = None,
                      progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[int, str]:
        """
        Nazywa klastry na podstawie zawartych w nich słów kluczowych.
        
        Args:
            df: DataFrame zawierający słowa kluczowe i etykiety klastrów
            max_keywords_per_cluster: Maksymalna liczba słów kluczowych do przesłania do API (None = wszystkie)
            progress_callback: Funkcja callback dla aktualizacji paska postępu (progress, status)
            
        Returns:
            Słownik mapujący ID klastrów na ich nazwy
        """
        if 'cluster' not in df.columns or 'keyword' not in df.columns:
            raise ValueError("DataFrame musi zawierać kolumny 'cluster' i 'keyword'")
        
        logger.info("Rozpoczynam nazywanie klastrów...")
        
        if progress_callback:
            progress_callback(0.0, "Przygotowanie do nazywania klastrów...")
        
        # Słownik na wyniki
        cluster_names = {}
        
        # Pobierz unikalne ID klastrów (bez -1, który oznacza szum)
        unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
        total_clusters = len(unique_clusters)
        
        # Wybór między tqdm a progress_callback
        if progress_callback:
            iterator = enumerate(unique_clusters)
        else:
            iterator = enumerate(tqdm(unique_clusters, desc="Nazywanie klastrów"))
        
        for idx, cluster_id in iterator:
            # Pobierz słowa kluczowe z klastra
            cluster_keywords = df[df['cluster'] == cluster_id]['keyword'].tolist()
            
            # Sortowanie według wolumenu (jeśli dostępny)
            if 'volume' in df.columns:
                cluster_data = df[df['cluster'] == cluster_id].sort_values(by='volume', ascending=False)
                cluster_keywords = cluster_data['keyword'].tolist()
            
            # Ogranicz liczbę słów kluczowych tylko jeśli podano limit i jest przekroczony
            if max_keywords_per_cluster is not None and len(cluster_keywords) > max_keywords_per_cluster:
                sample_keywords = cluster_keywords[:max_keywords_per_cluster]
                keyword_info = f"Wybraną próbkę {max_keywords_per_cluster} z {len(cluster_keywords)} słów kluczowych"
            else:
                sample_keywords = cluster_keywords
                keyword_info = f"Wszystkie {len(cluster_keywords)} słów kluczowych"
            
            # Przygotuj prompt dla modelu z głębszą analizą
            prompt = f"""
            Poniżej znajduje się lista powiązanych słów kluczowych, które zostały algorytmicznie zgrupowane:
            
            {', '.join(sample_keywords)}
            
            Kontekst: Ta lista zawiera {keyword_info} w tym klastrze.
            
            Na podstawie tej listy:
            1. Zidentyfikuj wspólny temat lub dziedzinę łączącą te słowa kluczowe.
            2. Przeanalizuj intencję użytkownika stojącą za tymi słowami (informacyjna, transakcyjna, itp.).
            3. Utwórz krótką (2-5 słów), opisową nazwę dla tego klastra tematycznego.
            4. Nazwa powinna być konkretna i precyzyjna, pozwalająca odróżnić ten klaster od innych.
            
            Odpowiedz TYLKO nazwą klastra, bez dodatkowego tekstu czy wyjaśnień.
            """
            
            try:
                # Wywołaj model Gemini
                response = self.model.generate_content(prompt)
                
                # Pobierz nazwę klastra
                cluster_name = response.text.strip()
                
                # Jeśli odpowiedź jest pusta, użyj domyślnej nazwy
                if not cluster_name:
                    cluster_name = f"Cluster {cluster_id}"
                
                # Zapisz nazwę klastra
                cluster_names[cluster_id] = cluster_name
                
                # Aktualizuj pasek postępu
                if progress_callback:
                    progress = (idx + 1) / total_clusters
                    status = f"Nazywanie klastra {idx + 1}/{total_clusters}: {cluster_name}"
                    progress_callback(progress, status)
                
                # Krótka przerwa, aby uniknąć limitów API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Błąd podczas nazywania klastra {cluster_id}: {e}")
                cluster_names[cluster_id] = f"Cluster {cluster_id}"
        
        logger.info(f"Zakończono nazywanie {len(cluster_names)} klastrów")
        
        if progress_callback:
            progress_callback(1.0, f"Zakończono nazywanie klastrów ({len(cluster_names)} klastrów)")
        
        return cluster_names
    
    def analyze_cluster_content(self, df: pd.DataFrame, cluster_id: int, cluster_name: str, max_keywords: Optional[int] = None) -> Dict[str, Any]:
        """
        Analizuje zawartość klastra, generując wnioski i rekomendacje.
        
        Args:
            df: DataFrame zawierający słowa kluczowe i etykiety klastrów
            cluster_id: ID klastra do analizy
            cluster_name: Nazwa klastra
            max_keywords: Maksymalna liczba słów kluczowych do przesłania do API (None = wszystkie)
            
        Returns:
            Słownik zawierający wyniki analizy
        """
        # Pobierz słowa kluczowe z klastra
        cluster_df = df[df['cluster'] == cluster_id]
        
        if len(cluster_df) == 0:
            return {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "keywords_count": 0,
                "total_volume": 0,
                "insights": "Brak słów kluczowych w klastrze",
                "content_strategy": "Nie dotyczy",
                "priority_score": 0,
                "priority_level": "Niski"
            }
        
        # Pobierz podstawowe statystyki
        keywords_count = len(cluster_df)
        total_volume = int(cluster_df['volume'].sum())
        avg_difficulty = round(float(cluster_df['difficulty'].mean()), 2)
        avg_cpc = round(float(cluster_df['cpc'].mean()), 2)
        
        # Określ intencje w klastrze - sprawdź nowy format intencji z Ahrefs
        intent_columns = ['branded', 'local', 'navigational', 'informational', 'commercial', 'transactional']
        available_intent_columns = [col for col in intent_columns if col in cluster_df.columns]
        
        if available_intent_columns:
            # Nowy format intencji z kolumn Ahrefs
            intent_distribution = {}
            total_keywords = len(cluster_df)
            
            for intent in available_intent_columns:
                count = cluster_df[intent].sum()  # Suma wartości True
                percentage = (count / total_keywords) * 100
                intent_distribution[intent] = {
                    'count': int(count),
                    'percentage': round(percentage, 2)
                }
            
            # Podsumowanie intencji dla prompta
            intent_summary = ", ".join([f"{intent.capitalize()}: {data['count']} ({data['percentage']}%)" 
                                       for intent, data in intent_distribution.items()])
            
            # Sprawdź, ile słów ma przypisaną więcej niż jedną intencję
            multiple_intents_count = cluster_df.apply(
                lambda row: sum(row[col] for col in available_intent_columns if col in row),
                axis=1
            )
            
            keywords_with_multiple_intents = (multiple_intents_count > 1).sum()
            multiple_intents_pct = round((keywords_with_multiple_intents / total_keywords) * 100, 2)
            
            intent_summary += f"\nUwaga: {keywords_with_multiple_intents} słów ({multiple_intents_pct}%) ma więcej niż jedną intencję"
        else:
            # Stary format intencji z pojedynczej kolumny 'intent'
            intent_distribution = cluster_df['intent'].value_counts().to_dict()
            intent_summary = ", ".join([f"{intent}: {count}" for intent, count in intent_distribution.items()])
        
        # Sortuj według wolumenu (najbardziej popularne na górze)
        cluster_df_sorted = cluster_df.sort_values(by='volume', ascending=False)
        
        # Używamy wszystkich słów kluczowych, chyba że podano limit
        if max_keywords is not None and len(cluster_df_sorted) > max_keywords:
            sample_keywords = cluster_df_sorted.head(max_keywords)
            keyword_info = f"Próbka {max_keywords} z {len(cluster_df_sorted)} słów kluczowych (sortowane wg wolumenu)"
        else:
            sample_keywords = cluster_df_sorted
            keyword_info = f"Wszystkie {len(cluster_df_sorted)} słów kluczowych w klastrze"
        
        # Wybierz kolumny do wyświetlenia
        display_columns = ['keyword', 'volume', 'difficulty', 'cpc']
        
        # Dodaj kolumny intencji, jeśli są dostępne
        if available_intent_columns:
            display_columns.extend(available_intent_columns)
        elif 'intent' in cluster_df.columns:
            display_columns.append('intent')
        
        # Przygotuj formatowaną tabelę słów kluczowych
        keywords_table = sample_keywords[display_columns].to_string(index=False)
        
        # Przygotuj dodatkowe informacje statystyczne o klastrze
        stats_summary = f"""
        Statystyki klastra:
        - Liczba słów kluczowych: {keywords_count}
        - Łączny wolumen: {total_volume}
        - Średnia trudność (KD): {avg_difficulty}
        - Średni CPC: ${avg_cpc}
        - Rozkład intencji: {intent_summary}
        """
        
        # Zmodyfikowany prompt z prośbą o kompleksową analizę w określonym formacie
        prompt = f"""
        # Analiza klastra słów kluczowych: "{cluster_name}"

        ## Dane klastra
        {stats_summary}

        ## Zawartość klastra
        {keyword_info}:

        {keywords_table}

        ## Zadanie
        Przeprowadź dogłębną analizę tego klastra słów kluczowych, biorąc pod uwagę wszystkie przedstawione dane. Przeanalizuj wzorce, intencje użytkowników i potencjał biznesowy.

        Przygotuj profesjonalną analizę SEO z następującymi sekcjami:

        1. INSIGHTS: 
           - Szczegółowa analiza tego, co te słowa kluczowe mówią o intencjach i zachowaniach wyszukiwania użytkowników
           - Identyfikacja wzorców i trendów
           - Analiza ścieżki użytkownika i cyklu zakupowego reprezentowanego przez te frazy
           - 4-6 zdań pogłębionej analizy

        2. CONTENT STRATEGY: 
           - Konkretne, praktyczne rekomendacje dotyczące treści, które skutecznie targetowałyby te słowa kluczowe
           - Sugestie typów treści, formatów i tematów
           - Strategia dotycząca nagłówków, meta opisów i struktury treści
           - 4-6 zdań praktycznych wskazówek

        3. PRIORITY LEVEL: 
           - Oceń priorytet tego klastra jako "High", "Medium" lub "Low" na podstawie:
             * Wolumenu wyszukiwań
             * Trudności konkurencyjnej
             * Intencji komercyjnej
             * Potencjału konwersji
           - Uzasadnij swoją ocenę konkretnymi czynnikami

        Wykorzystaj całą swoją wiedzę o SEO, marketingu treści i zachowaniach użytkowników, aby stworzyć wartościową analizę.

        Format odpowiedzi:
        INSIGHTS:
        [Twoja pogłębiona analiza]

        CONTENT STRATEGY:
        [Twoje konkretne rekomendacje]

        PRIORITY LEVEL:
        [High/Medium/Low + krótkie uzasadnienie]
        """
        
        try:
            # Wywołaj model Gemini
            response = self.model.generate_content(prompt)
            
            # Pobierz wyniki analizy
            analysis_text = response.text.strip()
            
            if not analysis_text:
                raise ValueError("Model zwrócił pustą odpowiedź")
            
            # Poszukaj sekcji w tekście za pomocą rozdzielania na sekcje
            insights = ""
            content_strategy = ""
            priority_level = "Medium"  # Domyślna wartość
            priority_justification = ""
            
            # Znajdź sekcje za pomocą rozdzielania tekstu
            if "INSIGHTS:" in analysis_text and "CONTENT STRATEGY:" in analysis_text:
                # Podziel tekst na sekcje
                insights_start = analysis_text.find("INSIGHTS:")
                content_start = analysis_text.find("CONTENT STRATEGY:")
                priority_start = analysis_text.find("PRIORITY LEVEL:")
                
                # Wyodrębnij sekcje (jeśli istnieją)
                if insights_start >= 0 and content_start > insights_start:
                    insights = analysis_text[insights_start + 9:content_start].strip()
                
                if content_start >= 0 and priority_start > content_start:
                    content_strategy = analysis_text[content_start + 17:priority_start].strip()
                elif content_start >= 0:
                    content_strategy = analysis_text[content_start + 17:].strip()
                    
                if priority_start >= 0:
                    priority_text = analysis_text[priority_start + 14:].strip().lower()
                    # Określ poziom priorytetu
                    if "high" in priority_text[:20]:
                        priority_level = "High"
                    elif "low" in priority_text[:20]:
                        priority_level = "Low"
                    else:
                        priority_level = "Medium"
                    
                    # Zapisz uzasadnienie
                    priority_justification = analysis_text[priority_start + 14:].strip()
            
            # Jeśli nie znaleziono sekcji, użyj całego tekstu
            if not insights and not content_strategy:
                insights = "Model nie wygenerował analizy w oczekiwanym formacie. Dostępna jest surowa odpowiedź."
                content_strategy = analysis_text
            
            # Oblicz priorytet na podstawie metryk i rekomendacji modelu
            priority_metrics = self._calculate_priority_score(
                total_volume, 
                avg_difficulty, 
                avg_cpc, 
                intent_distribution
            )
            
            # Mapowanie poziomu priorytetu na wartość liczbową
            priority_mapping = {"High": 3, "Medium": 2, "Low": 1}
            model_priority = priority_mapping.get(priority_level, 2)
            
            # Łączny wynik priorytetu (waga modelu większa dla lepszej jakości)
            combined_priority = (priority_metrics + model_priority * 2) / 3
            
            # Określenie poziomu priorytetu
            if combined_priority >= 2.5:
                priority_level = "Wysoki"
            elif combined_priority >= 1.5:
                priority_level = "Średni"
            else:
                priority_level = "Niski"
            
            # Przygotuj wyniki analizy
            analysis_results = {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "keywords_count": keywords_count,
                "total_volume": total_volume,
                "avg_difficulty": avg_difficulty,
                "avg_cpc": avg_cpc,
                "intent_distribution": intent_distribution,
                "insights": insights,
                "content_strategy": content_strategy,
                "priority_justification": priority_justification,
                "priority_score": round(combined_priority, 2),
                "priority_level": priority_level
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Błąd podczas analizy klastra {cluster_id}: {e}")
            return {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "keywords_count": keywords_count,
                "total_volume": total_volume,
                "avg_difficulty": avg_difficulty,
                "avg_cpc": avg_cpc,
                "intent_distribution": intent_distribution,
                "insights": "Wystąpił błąd podczas analizy klastra. Spróbuj ręcznej analizy.",
                "content_strategy": "Nie można wygenerować rekomendacji z powodu błędu: " + str(e),
                "priority_score": self._calculate_priority_score(total_volume, avg_difficulty, avg_cpc, intent_distribution),
                "priority_level": "Średni"
            }
    
    def _calculate_priority_score(self, volume: int, difficulty: float, cpc: float, intent_distribution: Dict[str, Any]) -> float:
        """
        Oblicza wynik priorytetu dla klastra na podstawie metryk.
        
        Args:
            volume: Łączny wolumen wyszukiwań
            difficulty: Średnia trudność słów kluczowych
            cpc: Średni koszt kliknięcia
            intent_distribution: Rozkład intencji w klastrze
            
        Returns:
            Wynik priorytetu (1-3)
        """
        # Normalizacja wolumenu (logarytmicznie, aby uwzględnić duże różnice)
        volume_score = min(3, max(1, np.log10(volume + 1) / 3))
        
        # Normalizacja trudności (odwrotnie proporcjonalna)
        difficulty_score = min(3, max(1, 3 - (difficulty / 100) * 2))
        
        # Normalizacja CPC
        cpc_score = min(3, max(1, cpc + 1))
        
        # Waga intencji (wyższy wynik dla intencji transakcyjnych i komercyjnych)
        intent_weight = 1.0
        
        # Sprawdź, czy mamy nowy czy stary format intencji
        if isinstance(next(iter(intent_distribution.values()), None), dict):
            # Nowy format - każda intencja ma słownik z count i percentage
            commercial_pct = intent_distribution.get('commercial', {}).get('percentage', 0) / 100
            transactional_pct = intent_distribution.get('transactional', {}).get('percentage', 0) / 100
            branded_pct = intent_distribution.get('branded', {}).get('percentage', 0) / 100
            
            # Waż intencje z naciskiem na transakcyjne i komercyjne
            intent_weight = 1.0 + 2.0 * (transactional_pct + commercial_pct) + branded_pct
        else:
            # Stary format - wartości liczbowe dla każdej intencji
            total_keywords = sum(intent_distribution.values())
            
            if total_keywords > 0:
                # Oblicz procent intencji transakcyjnych i komercyjnych
                transaction_pct = (intent_distribution.get('transactional', 0) / total_keywords)
                commercial_pct = (intent_distribution.get('commercial', 0) / total_keywords)
                
                # Wyższy wynik dla klastrów z większą liczbą intencji transakcyjnych i komercyjnych
                intent_weight = 1.0 + 2.0 * (transaction_pct + commercial_pct)
        
        # Łączny wynik priorytetu
        priority_score = (
            volume_score * VOLUME_WEIGHT +
            difficulty_score * KD_WEIGHT +
            cpc_score * CPC_WEIGHT
        ) * intent_weight
        
        # Normalizacja do zakresu 1-3
        normalized_score = min(3, max(1, priority_score))
        
        return normalized_score
    
    def process_all_clusters(self, df: pd.DataFrame, cluster_names: Dict[int, str], max_keywords: Optional[int] = None,
                             progress_callback: Optional[Callable[[float, str], None]] = None) -> List[Dict[str, Any]]:
        """
        Przetwarza wszystkie klastry, generując analizę dla każdego z nich.
        
        Args:
            df: DataFrame zawierający słowa kluczowe i etykiety klastrów
            cluster_names: Słownik mapujący ID klastrów na ich nazwy
            max_keywords: Maksymalna liczba słów kluczowych do analizy dla każdego klastra (None = wszystkie)
            progress_callback: Funkcja callback dla aktualizacji paska postępu (progress, status)
            
        Returns:
            Lista słowników zawierających wyniki analizy dla każdego klastra
        """
        logger.info("Rozpoczynam analizę wszystkich klastrów...")
        
        # Lista na wyniki analizy
        all_analyses = []
        
        # Pobierz unikalne ID klastrów (bez -1, który oznacza szum)
        unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
        total_clusters = len(unique_clusters)
        
        # Wybór między tqdm a progress_callback
        if progress_callback:
            iterator = enumerate(unique_clusters)
        else:
            iterator = enumerate(tqdm(unique_clusters, desc="Analizowanie klastrów"))
        
        for idx, cluster_id in iterator:
            cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            
            # Analizuj klaster
            analysis = self.analyze_cluster_content(df, cluster_id, cluster_name, max_keywords)
            
            # Dodaj analizę do wyników
            all_analyses.append(analysis)
            
            # Aktualizuj pasek postępu
            if progress_callback:
                progress = (idx + 1) / total_clusters
                status = f"Analiza klastra {idx + 1}/{total_clusters}: {cluster_name}"
                progress_callback(progress, status)
            
            # Krótka przerwa, aby uniknąć limitów API
            time.sleep(1)
        
        # Sortuj klastry według priorytetu (malejąco)
        all_analyses = sorted(all_analyses, key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"Zakończono analizę {len(all_analyses)} klastrów")
        
        if progress_callback:
            progress_callback(1.0, f"Analiza klastrów zakończona ({len(all_analyses)} klastrów)")
        
        return all_analyses