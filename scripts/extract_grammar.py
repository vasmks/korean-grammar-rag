import json
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "grammar_entries.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_feat_value(feats, name):
    for feat in as_list(feats):
        if isinstance(feat, dict) and feat.get("att") == name:
            return feat.get("val")
    return None


grammar_entries = []

for file_path in RAW_DIR.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["LexicalResource"]["Lexicon"]["LexicalEntry"]

    for entry in as_list(entries):
        lexical_unit = get_feat_value(
            entry.get("feat"),
            "lexicalUnit"
        )

        part_of_speech = get_feat_value(
            entry.get("feat"),
            "partOfSpeech"
        )

        if (
            lexical_unit == "문법‧표현"
            or part_of_speech in {"어미", "조사"}
        ):
            grammar_entries.append(entry)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        grammar_entries,
        f,
        ensure_ascii=False,
        indent=2
    )


print(f"Extracted {len(grammar_entries)} grammar-related entries.")
print(f"Saved to: {OUTPUT_FILE}")