# analysis_engine.py
import spacy
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer

from src.emotions import detecter_emotions
from src.aspects import analyse_aspects, ASPECTS

nlp = spacy.load("fr_core_news_md")
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
vader = SentimentIntensityAnalyzer()


# --- SENTIMENT GLOBAL ---
def analyse_document(text):
    tb_score = tb(text).sentiment[0]
    vader_score = vader.polarity_scores(text)["compound"]
    final_score = (tb_score + vader_score) / 2

    coherence = round(1 - abs(tb_score - vader_score), 3)

    return {
        "score": round(final_score, 3),
        "label": classifier(final_score),
        "tb_score": round(tb_score, 3),
        "vader_score": round(vader_score, 3),
        "coherence": coherence
    }


def classifier(score):
    if score > 0.6: return "Très Positif"
    if score > 0.2: return "Positif"
    if score > -0.2: return "Neutre"
    if score > -0.6: return "Négatif"
    return "Très Négatif"


# --- SENTIMENT PAR PHRASE ---
def analyse_phrases(text):
    doc = nlp(text)
    results = []

    for phrase in doc.sents:
        p = str(phrase).strip()
        score = tb(p).sentiment[0]
        results.append({
            "phrase": p,
            "score": round(score, 3),
            "label": classifier(score)
        })

    return results


# --- DETECTION DE CONTRADICTIONS ---
def detect_contradictions(phrases):
    pos = [p for p in phrases if p["score"] > 0.3]
    neg = [p for p in phrases if p["score"] < -0.3]

    contradictions = []
    if pos and neg:
        contradictions = pos + neg

    return contradictions


# --- RESUME GLOBAL ---
def build_summary(doc_sent, aspects, emotions):
    # Aspect le plus fort/faible
    valid_aspects = {k: v for k, v in aspects.items() if v is not None}

    best_aspect = max(valid_aspects, key=valid_aspects.get) if valid_aspects else None
    worst_aspect = min(valid_aspects, key=valid_aspects.get) if valid_aspects else None

    # Émotion dominante
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "best_aspect": best_aspect,
        "worst_aspect": worst_aspect,
        "dominant_emotion": dominant_emotion,
        "coherence": doc_sent["coherence"]
    }


# --- PIPELINE COMPLET ---
def analyse_complete(text):
    doc = analyse_document(text)
    phrases = analyse_phrases(text)
    aspects = analyse_aspects(text)
    emotions = detecter_emotions(text)
    contradictions = detect_contradictions(phrases)

    summary = build_summary(doc, aspects, emotions)

    return {
        "document": doc,
        "phrases": phrases,
        "aspects": aspects,
        "emotions": emotions,
        "contradictions": contradictions,
        "summary": summary
    }

