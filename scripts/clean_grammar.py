import json
import html
from pathlib import Path

INPUT_FILE = Path("data/processed/grammar_entries.json")
OUTPUT_FILE = Path("data/processed/grammar_clean.json")


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def feat_value(feats, name):
    for feat in as_list(feats):
        if isinstance(feat, dict) and feat.get("att") == name:
            return feat.get("val")
    return None


def feat_values(feats, name):
    return [
        feat.get("val")
        for feat in as_list(feats)
        if isinstance(feat, dict)
        and feat.get("att") == name
        and feat.get("val") is not None
    ]


def get_written_form(entry):
    for lemma in as_list(entry.get("Lemma")):
        if not isinstance(lemma, dict):
            continue

        written_form = feat_value(
            lemma.get("feat"),
            "writtenForm"
        )

        if written_form:
            return written_form

    return None


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = json.load(f)


clean_entries = []

for entry in entries:
    grammar = get_written_form(entry)

    if not grammar:
        continue

    part_of_speech = feat_value(
        entry.get("feat"),
        "partOfSpeech"
    )

    lexical_unit = feat_value(
        entry.get("feat"),
        "lexicalUnit"
    )

    senses = as_list(entry.get("Sense"))

    for sense_index, sense in enumerate(senses, start=1):
        if not isinstance(sense, dict):
            continue

        sense_feats = sense.get("feat", [])

        definition_ko = feat_value(
            sense_feats,
            "definition"
        )

        usage = feat_value(
            sense_feats,
            "annotation"
        )

        definition_en = None

        for equivalent in as_list(sense.get("Equivalent")):
            if not isinstance(equivalent, dict):
                continue

            feats = equivalent.get("feat", [])

            if feat_value(feats, "language") == "영어":
                definition_en = feat_value(
                    feats,
                    "definition"
                )
                break

        examples = []

        for example_item in as_list(sense.get("SenseExample")):
            if not isinstance(example_item, dict):
                continue

            examples.extend(
                html.unescape(text.strip())
                for text in feat_values(
                    example_item.get("feat", []),
                    "example"
                )
            )

        related = []

        for relation in as_list(sense.get("SenseRelation")):
            if not isinstance(relation, dict):
                continue

            feats = relation.get("feat", [])

            lemma = feat_value(feats, "lemma")
            relation_type = feat_value(feats, "type")

            if lemma:
                related.append({
                    "grammar": lemma,
                    "type": relation_type
                })

        clean_entries.append({
            "id": entry.get("val"),
            "sense_id": sense.get("val"),
            "sense_number": sense_index,
            "grammar": grammar,
            "lexical_unit": lexical_unit,
            "part_of_speech": part_of_speech,
            "definition_ko": html.unescape(definition_ko or ""),
            "definition_en": html.unescape(definition_en or ""),
            "usage": html.unescape(usage or ""),
            "examples": examples,
            "related": related,
            "source": "Korean Basic Dictionary"
        })


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        clean_entries,
        f,
        ensure_ascii=False,
        indent=2
    )


print(f"Original entries: {len(entries)}")
print(f"Cleaned grammar senses: {len(clean_entries)}")
print(f"Saved to: {OUTPUT_FILE}")