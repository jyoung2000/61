"""Language-code normalization shared across the pipeline.

Every internal consumer — the translator router, NMT model-ID templates
(``staka/fugumt-{src}-{tgt}``, ``Helsinki-NLP/opus-mt-{src}-{tgt}``), the
Flores-200 map, the transcript polisher's CJK rules and the pipeline's
"non-English → auto-translate" check — keys on ISO 639-1 codes ("ja", "en").
But the language can enter as a full English name: whisper.cpp reports
``"language": "japanese"`` and users may type "Japanese" in settings. An
unnormalized name silently breaks all of those lookups — the observed failure
was an invalid HF repo id ``staka/fugumt-japanese-en`` aborting translation.

``normalize_lang_code`` maps any spelling (full name, ISO 639-2/3, common
alias, region-tagged variant) to the ISO 639-1 code the rest of the codebase
expects. Unknown values pass through lowercased so novel-but-valid codes
still work.
"""

# Whisper's language inventory (name → ISO 639-1), plus ISO 639-2/3 and
# frequent aliases. Region-tagged Chinese variants are preserved (the Flores
# map distinguishes Hans/Hant), all else collapses to the bare code.
_NAME_TO_ISO = {
    # ── Whisper full names ──
    "english": "en", "chinese": "zh", "german": "de", "spanish": "es",
    "russian": "ru", "korean": "ko", "french": "fr", "japanese": "ja",
    "portuguese": "pt", "turkish": "tr", "polish": "pl", "catalan": "ca",
    "dutch": "nl", "arabic": "ar", "swedish": "sv", "italian": "it",
    "indonesian": "id", "hindi": "hi", "finnish": "fi", "vietnamese": "vi",
    "hebrew": "he", "ukrainian": "uk", "greek": "el", "malay": "ms",
    "czech": "cs", "romanian": "ro", "danish": "da", "hungarian": "hu",
    "tamil": "ta", "norwegian": "no", "thai": "th", "urdu": "ur",
    "croatian": "hr", "bulgarian": "bg", "lithuanian": "lt", "latin": "la",
    "maori": "mi", "malayalam": "ml", "welsh": "cy", "slovak": "sk",
    "telugu": "te", "persian": "fa", "latvian": "lv", "bengali": "bn",
    "serbian": "sr", "azerbaijani": "az", "slovenian": "sl", "kannada": "kn",
    "estonian": "et", "macedonian": "mk", "breton": "br", "basque": "eu",
    "icelandic": "is", "armenian": "hy", "nepali": "ne", "mongolian": "mn",
    "bosnian": "bs", "kazakh": "kk", "albanian": "sq", "swahili": "sw",
    "galician": "gl", "marathi": "mr", "punjabi": "pa", "sinhala": "si",
    "khmer": "km", "shona": "sn", "yoruba": "yo", "somali": "so",
    "afrikaans": "af", "occitan": "oc", "georgian": "ka", "belarusian": "be",
    "tajik": "tg", "sindhi": "sd", "gujarati": "gu", "amharic": "am",
    "yiddish": "yi", "lao": "lo", "uzbek": "uz", "faroese": "fo",
    "haitian creole": "ht", "haitian": "ht", "pashto": "ps", "turkmen": "tk",
    "nynorsk": "nn", "maltese": "mt", "sanskrit": "sa", "luxembourgish": "lb",
    "myanmar": "my", "burmese": "my", "tibetan": "bo", "tagalog": "tl",
    "malagasy": "mg", "assamese": "as", "tatar": "tt", "hawaiian": "haw",
    "lingala": "ln", "hausa": "ha", "bashkir": "ba", "javanese": "jv",
    "sundanese": "su", "cantonese": "yue", "flemish": "nl",
    "castilian": "es", "moldavian": "ro", "moldovan": "ro", "valencian": "ca",
    "letzeburgesch": "lb", "pushto": "ps", "panjabi": "pa", "sinhalese": "si",
    # ── ISO 639-2/3 + common aliases ──
    "jpn": "ja", "jp": "ja", "eng": "en", "kor": "ko", "kr": "ko",
    "zho": "zh", "chi": "zh", "cn": "zh", "deu": "de", "ger": "de",
    "fra": "fr", "fre": "fr", "spa": "es", "por": "pt", "rus": "ru",
    "ita": "it", "nld": "nl", "dut": "nl", "pol": "pl", "tur": "tr",
    "ara": "ar", "hin": "hi", "vie": "vi", "tha": "th", "ind": "id",
    "ukr": "uk", "ces": "cs", "cze": "cs", "swe": "sv", "dan": "da",
    "fin": "fi", "nor": "no", "ell": "el", "gre": "el", "heb": "he",
    "iw": "he", "fil": "tl", "filipino": "tl",
}

# Region-tagged codes that must NOT collapse to the bare language code
# (the Flores map distinguishes Simplified/Traditional Chinese).
_KEEP_REGIONAL = {"zh-cn", "zh-tw", "zh-hk", "zh-hans", "zh-hant"}


def normalize_lang_code(lang: str) -> str:
    """Normalize any language spelling to the ISO 639-1 code the pipeline
    keys on. Passthroughs: empty / "auto" / already-ISO / unknown values
    (lowercased). Never raises.
    """
    s = (lang or "").strip().lower()
    if not s or s == "auto":
        return s
    if s in _KEEP_REGIONAL:
        return s
    if s in _NAME_TO_ISO:
        return _NAME_TO_ISO[s]
    # Region-tagged variants of known names/codes ("japanese-jp", "pt_BR")
    # collapse to the base language.
    base = s.replace("_", "-").split("-")[0]
    if base in _NAME_TO_ISO:
        return _NAME_TO_ISO[base]
    if base != s and len(base) == 2:
        return base
    return s
