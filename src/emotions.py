import math

EMOTIONS = {
    "joie": ["excellent", "super", "satisfait", "parfait", "impeccable"],
    "colère": ["frustré", "furieux", "énervé"],
    "tristesse": ["déçu", "triste"],
    "peur": ["inquiet", "peur"],
    "surprise": ["surprenant", "étonnant"],
    "degout": ["horrible", "dégoûtant", "bof"]
}

def detecter_emotions(text):
    text = text.lower()
    scores = {e: 0 for e in EMOTIONS}

    for emotion, words in EMOTIONS.items():
        for w in words:
            if w in text:
                scores[emotion] += 1

    total = sum(scores.values())
    if total > 0:
        for e in scores:
            scores[e] = round(scores[e] / total, 2)

    return scores
