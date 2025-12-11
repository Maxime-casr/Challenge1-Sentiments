import spacy
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer

nlp = spacy.load("fr_core_news_md")
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

ASPECTS = {
    "produit": ["qualité", "performance", "son", "caméra", "application"],
    "prix": ["coût", "cher", "prix", "valeur"],
    "service": ["support", "client", "SAV"],
    "livraison": ["livraison", "rapide", "endommagé", "colis"]
}

def analyse_aspects(text):
    doc = nlp(text)
    result = {a: [] for a in ASPECTS}

    for sent in doc.sents:
        s = str(sent)
        for aspect, keywords in ASPECTS.items():
            if any(word in s.lower() for word in keywords):
                sent_score = tb(s).sentiment[0]
                result[aspect].append({"phrase": s, "score": sent_score})

    # Moyenne par aspect
    final = {}
    for a, values in result.items():
        if values:
            avg = sum(v["score"] for v in values) / len(values)
            final[a] = round(avg, 3)
        else:
            final[a] = None

    return final
