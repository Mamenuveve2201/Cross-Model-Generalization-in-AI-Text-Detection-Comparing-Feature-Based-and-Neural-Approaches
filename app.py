import gradio as gr
import joblib
import pandas as pd
import numpy as np
import re
import math
import collections
import string
from sklearn.preprocessing import StandardScaler

# ── LOAD MODELS ───────────────────────────────────────────────────────
lr_models  = [joblib.load(f"models/logistic/fold{i}_model.pkl") for i in range(1, 6)]
rf_models  = [joblib.load(f"models/random_forest/fold{i}_model.pkl") for i in range(1, 6)]
xgb_models = [joblib.load(f"models/xgboost/fold{i}_model.pkl") for i in range(1, 6)]

# Load training data for scaler fitting
train_data = pd.read_csv("data/raw/features_all.csv")
META_COLS  = ["essay_id", "text", "label", "model_source"]
FEAT_COLS  = [c for c in train_data.columns if c not in META_COLS]
scaler     = StandardScaler()
scaler.fit(train_data[FEAT_COLS].fillna(0))

# ── FEATURE EXTRACTION ────────────────────────────────────────────────
STOPWORDS = set(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are "
    "was were be been being have has had having do does did doing a an the "
    "and but if or because as until while of at by for with about against "
    "between into through during before after above below to from up down "
    "in out on off over under again further then once here there when where "
    "why how all both each few more most other some such no nor not only own "
    "same so than too very s t can will just don should now d ll m o re ve "
    "y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn "
    "shan shouldn wasn weren won wouldn".split()
)

PASSIVE_PATTERNS = [
    r"\b(is|are|was|were|be|been|being)\s+\w+ed\b",
    r"\b(is|are|was|were)\s+being\s+\w+ed\b",
    r"\b(has|have|had)\s+been\s+\w+ed\b",
]

def count_syllables(word):
    word = word.lower().strip("'")
    if not word:
        return 0
    count = len(re.findall(r"[aeiou]+", word))
    if word.endswith("e") and len(word) > 2 and count > 1:
        count -= 1
    return max(1, count)

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def extract_features(text):
    if not text or not text.strip():
        return {f: 0.0 for f in FEAT_COLS}

    words      = re.findall(r"[a-zA-Z']+", text.lower())
    alpha      = [w for w in words if re.match(r"^[a-z']+$", w)]
    n          = max(len(alpha), 1)
    freq       = collections.Counter(alpha)
    V          = max(len(freq), 1)
    sentences  = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if s.strip()]
    n_sents    = max(len(sentences), 1)
    sent_lens  = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences]
    text_lower = text.lower()

    feats = {}

    # Stylometric
    feats["type_token_ratio"]     = V / n
    feats["unique_words_ratio"]   = V / n
    feats["vocab_richness"]       = V / math.sqrt(n)
    M1 = n; M2 = sum(v**2 for v in freq.values())
    feats["yules_k"]              = 10**4 * (M2 - M1) / (M1**2) if M1 > 0 else 0
    wlens = [len(w) for w in alpha]
    feats["avg_word_len"]         = np.mean(wlens) if wlens else 0
    feats["word_length_variance"] = float(np.var(wlens)) if wlens else 0
    feats["syllables_per_word"]   = float(np.mean([count_syllables(w) for w in alpha])) if alpha else 0
    feats["long_word_ratio"]      = sum(1 for w in alpha if len(w) > 7) / n
    feats["short_word_ratio"]     = sum(1 for w in alpha if len(w) < 4) / n
    content = [w for w in alpha if w not in STOPWORDS]
    feats["lexical_density"]      = len(content) / n
    hapax = sum(1 for c in freq.values() if c == 1)
    feats["hapax_ratio"]          = hapax / V
    feats["avg_word_length_chars"]= np.mean(wlens) if wlens else 0

    # Semantic (Xia et al. key features)
    passive = sum(len(re.findall(p, text_lower)) for p in PASSIVE_PATTERNS)
    feats["passive_voice_freq"]         = passive / n
    active = len(re.findall(r"\b(is|are|am|was|were|do|does|did|have|has|had)\b", text_lower))
    feats["active_voice_freq"]          = max(0, active - passive) / n
    feats["past_tense_freq"]            = len(re.findall(r"\b\w+ed\b|\b(was|were|had|did)\b", text_lower)) / n
    feats["present_tense_freq"]         = len(re.findall(r"\b(is|are|am|has|have|do|does)\b", text_lower)) / n
    feats["future_tense_freq"]          = len(re.findall(r"\b(will|shall|going to|gonna)\b", text_lower)) / n
    feats["first_person_freq"]          = len(re.findall(r"\b(i|me|my|mine|myself|we|us|our)\b", text, re.I)) / n
    feats["second_person_freq"]         = len(re.findall(r"\b(you|your|yours|yourself)\b", text, re.I)) / n
    feats["third_person_freq"]          = len(re.findall(r"\b(he|she|it|they|them|their)\b", text, re.I)) / n

    # Statistical
    mean_sl = np.mean(sent_lens); std_sl = np.std(sent_lens)
    denom = std_sl + mean_sl
    feats["burstiness"]               = (std_sl - mean_sl) / denom if denom > 0 else 0
    freq2 = collections.Counter(alpha)
    total = sum(freq2.values())
    feats["entropy"]                  = -sum((c/total)*math.log2(c/total) for c in freq2.values() if c > 0) if total > 0 else 0
    bigrams  = get_ngrams(alpha, 2)
    feats["repetition_rate"]          = (1 - len(set(bigrams))/len(bigrams)) if bigrams else 0
    trigrams = get_ngrams(alpha, 3)
    feats["trigram_repetition"]       = (1 - len(set(trigrams))/len(trigrams)) if trigrams else 0
    first_words = [s.split()[0].lower() for s in sentences if s.split()]
    feats["sentence_start_diversity"] = len(set(first_words))/len(first_words) if first_words else 1.0
    feats["sent_len_mean"]            = float(mean_sl)
    feats["sent_len_std"]             = float(std_sl)
    feats["sent_len_min"]             = float(min(sent_lens)) if sent_lens else 0
    feats["sent_len_max"]             = float(max(sent_lens)) if sent_lens else 0
    feats["punctuation_density"]      = sum(1 for ch in text if ch in string.punctuation) / max(len(text), 1)
    feats["question_freq"]            = sum(1 for s in sentences if s.strip().endswith("?")) / n_sents

    # POS approximations
    feats["noun_freq"]          = len(re.findall(r"\b(the|a|an)\s+\w+", text_lower)) / n
    feats["verb_freq"]          = len(re.findall(r"\b(is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|shall|must|can)\b", text_lower)) / n
    feats["adj_freq"]           = len(re.findall(r"\b\w+(ful|less|ous|ive|al|ible|able|ic)\b", text_lower)) / n
    feats["adv_freq"]           = len(re.findall(r"\b\w+ly\b", text_lower)) / n
    feats["pos_pronoun_ratio"]  = len(re.findall(r"\b(i|me|my|you|he|she|it|we|they|them|their|his|her|its|our)\b", text_lower)) / n
    feats["pos_det_ratio"]      = len(re.findall(r"\b(the|a|an|this|that|these|those|my|your|his|her|its|our|their)\b", text_lower)) / n
    feats["pos_conj_ratio"]     = len(re.findall(r"\b(and|but|or|nor|for|yet|so|although|because|since|while|if|when|though)\b", text_lower)) / n
    feats["avg_dependency_depth"] = float(np.mean(wlens)) / 5 if wlens else 0
    feats["bigram_repetition"]  = feats["repetition_rate"]
    feats["noun_phrase_density"]= feats["noun_freq"]
    feats["named_entity_ratio"] = len(re.findall(r"\b[A-Z][a-z]+\b", text)) / n

    # Fill any missing features with 0
    for f in FEAT_COLS:
        if f not in feats:
            feats[f] = 0.0

    return feats

# ── TOP FEATURES FOR DISPLAY ──────────────────────────────────────────
TOP_FEATURES = [
    "syllables_per_word", "avg_word_len", "long_word_ratio",
    "hapax_ratio", "lexical_density", "word_length_variance",
    "type_token_ratio", "repetition_rate", "pos_pronoun_ratio",
    "short_word_ratio", "burstiness", "entropy",
    "first_person_freq", "passive_voice_freq", "sentence_start_diversity"
]

# ── PREDICT FUNCTION ──────────────────────────────────────────────────
def predict(text):
    if not text or len(text.strip()) < 50:
        return "⚠️ Please enter at least 50 characters.", "", ""

    feats = extract_features(text)
    feat_vec = pd.DataFrame([feats])[FEAT_COLS].fillna(0)
    feat_scaled = scaler.transform(feat_vec)

    # Get predictions from all 5 folds, average them
    lr_probs  = np.mean([m.predict_proba(feat_scaled)[0][1] for m in lr_models])
    rf_probs  = np.mean([m.predict_proba(feat_scaled)[0][1] for m in rf_models])
    xgb_probs = np.mean([m.predict_proba(feat_scaled)[0][1] for m in xgb_models])

    ensemble  = np.mean([lr_probs, rf_probs, xgb_probs])
    votes     = sum([lr_probs > 0.5, rf_probs > 0.5, xgb_probs > 0.5])

    # Verdict
    if ensemble > 0.5:
        verdict = f"🤖 AI-GENERATED ({votes}/3 models agree)"
        verdict_color = "red"
    else:
        verdict = f"✅ HUMAN-WRITTEN ({3 - votes}/3 models agree)"
        verdict_color = "green"

    # Model breakdown
    breakdown = f"""
**Ensemble Verdict: {verdict}**
---
| Model | AI Probability | Prediction |
|---|---|---|
| Logistic Regression | {lr_probs:.1%} | {"🤖 AI" if lr_probs > 0.5 else "✅ Human"} |
| Random Forest | {rf_probs:.1%} | {"🤖 AI" if rf_probs > 0.5 else "✅ Human"} |
| XGBoost | {xgb_probs:.1%} | {"🤖 AI" if xgb_probs > 0.5 else "✅ Human"} |
| **Ensemble** | **{ensemble:.1%}** | **{"🤖 AI" if ensemble > 0.5 else "✅ Human"}** |
"""

    # Feature values
    feat_display = "**Top Feature Values for This Text:**\n\n"
    feat_display += "| Feature | Value | What It Means |\n|---|---|---|\n"
    explanations = {
        "syllables_per_word":        "AI uses more complex words",
        "avg_word_len":              "AI uses longer words on average",
        "long_word_ratio":           "AI uses more words >7 chars",
        "hapax_ratio":               "Ratio of words used only once",
        "lexical_density":           "Content words vs total words",
        "word_length_variance":      "How varied word lengths are",
        "type_token_ratio":          "Vocabulary diversity",
        "repetition_rate":           "How often word pairs repeat",
        "pos_pronoun_ratio":         "Humans use more pronouns",
        "short_word_ratio":          "Humans use more short words",
        "burstiness":                "Variation in sentence lengths",
        "entropy":                   "Randomness of word distribution",
        "first_person_freq":         "Humans use I/we/my more",
        "passive_voice_freq":        "Use of passive constructions",
        "sentence_start_diversity":  "Variety in how sentences begin",
    }
    for f in TOP_FEATURES:
        val = feats.get(f, 0)
        exp = explanations.get(f, "")
        feat_display += f"| {f} | {val:.4f} | {exp} |\n"

    return breakdown, feat_display

# ── GRADIO UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="AI Text Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 AI Text Detector
    ### Cross-Model AI Text Detection using Linguistic Features
    Detects whether text was written by a human or generated by AI (GPT-4o, Claude, Gemini, LLaMA, Qwen).
    Uses 41 hand-crafted linguistic features with Leave-One-Model-Out cross-validation.
    
    *Built by Melissa Amenuveve | CAP5638 Pattern Recognition | Florida State University*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Paste your text here",
                placeholder="Enter at least 50 characters...",
                lines=10
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 Analyze Text", variant="primary", size="lg")
                clear_btn  = gr.Button("🗑️ Clear", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 How it works")
            gr.Markdown("""
            1. Extracts **41 linguistic features** from your text
            2. Runs **3 ML models** trained with LOMO-CV
            3. Returns **ensemble prediction** + confidence
            4. Shows **top features** that influenced the decision
            
            **Models trained on:**
            - GPT-4o, Claude, Gemini, LLaMA, Qwen
            - 5,981 samples total
            - Leave-One-Model-Out evaluation
            """)

    with gr.Row():
        with gr.Column():
            result_output  = gr.Markdown(label="Results")
        with gr.Column():
            feature_output = gr.Markdown(label="Feature Analysis")

    submit_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[result_output, feature_output]
    )
    clear_btn.click(lambda: ("", "", ""), outputs=[text_input, result_output, feature_output])

    gr.Markdown("""
    ---
    **Research context:** This detector was built as part of a study on cross-model generalization in AI text detection.
    Feature-based classical ML (LR, RF, XGBoost) was compared against neural baselines (RoBERTa-OpenAI, ChatGPT Detector).
    Results show feature-based models achieve 99%+ F1 vs ~18% F1 for neural baselines on unseen models.
    """)

demo.launch()