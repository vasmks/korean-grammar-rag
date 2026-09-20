import json
from collections import Counter
from pathlib import Path
from unittest import TestCase, skipUnless

import chromadb
import pymupdf

from app.topik_preview import PDF_DIR, topik_preview_service
from app.topik_retriever import DATA_FILE, TopikRetriever
from scripts.build_topik_retrieval_units import CONTEXTS_FILE, build_units
from scripts.generate_topik_crop_metadata import (
    compose_shared_visual_preview_regions,
    infer_independent_visual_starts,
    infer_shared_visual_starts,
    refine_visual_preview_regions,
)
from scripts.parse_topik_groups import parse_exam
from scripts.topik_page_extractor import (
    assess_native_structure,
    choose_auto_extraction,
    text_is_usable,
)
from scripts.topik_sources import DEFAULT_MANIFEST, load_sources


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROP_METADATA_FILE = PROJECT_ROOT / "data/topik/processed/topik_crop_metadata.json"
GROUPS_FILE = PROJECT_ROOT / "data/topik/processed/topik_reading_groups.json"
CHROMA_DIR = PROJECT_ROOT / "data/chroma_topik"


def make_page(lines, extraction_method="native"):
    return {
        "exam_number": 99,
        "source_file": "synthetic.pdf",
        "page": 1,
        "extraction_method": extraction_method,
        "lines": [
            {
                "text": text,
                "bbox": [50.0, y, 500.0, y + 15.0],
                "confidence": 0.9 if extraction_method == "ocr" else None,
            }
            for text, y in lines
        ],
    }


def make_native_candidate(lines):
    return {
        "page": 1,
        "text": "\n".join(lines),
        "lines": [{"text": text} for text in lines],
    }


class TopikStructuralParserTests(TestCase):
    def test_instruction_scope_produces_one_independent_unit_per_question(self):
        records = [
            make_page(
                [
                    ("※ [9~12] 다음 밑줄 친 부분과 의미가 가장 비슷한 것을 고르십시오.", 100),
                    ("9. 아홉 번째 문제", 150),
                    ("10. 열 번째 문제", 250),
                    ("11. 열한 번째 문제", 350),
                    ("12. 열두 번째 문제", 450),
                ]
            )
        ]
        scopes = parse_exam(records)
        units, contexts = build_units(scopes)
        self.assertEqual([unit["question_numbers"] for unit in units], [[9], [10], [11], [12]])
        self.assertEqual(contexts, [])

    def test_shared_context_is_attached_to_each_dependent_question(self):
        records = [
            make_page(
                [
                    ("※ [19~20] 다음 글을 읽고 물음에 답하십시오.", 100),
                    ("두 문제에 필요한 공동 지문입니다.", 140),
                    ("19. 첫 번째 물음", 300),
                    ("20. 두 번째 물음", 430),
                ],
                extraction_method="ocr",
            )
        ]
        scopes = parse_exam(records)
        units, contexts = build_units(scopes)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(units[0]["shared_context_id"], units[1]["shared_context_id"])
        self.assertIn("공동 지문", units[0]["text"])
        self.assertIn("공동 지문", units[1]["text"])
        self.assertNotIn("두 번째 물음", units[0]["question_text"])

    def test_missing_ocr_anchor_uses_clear_neighboring_layout_boundary(self):
        records = [
            make_page(
                [
                    ("※ [5~8] 다음 그림을 보고 물음에 답하십시오.", 100),
                    ("5. 첫 문제", 180),
                    ("6", 280),
                    ("일곱 번째 문제의 OCR 숫자가 없습니다.", 380),
                    ("8. 마지막 문제", 480),
                ],
                extraction_method="ocr",
            )
        ]
        scopes = parse_exam(records)
        units, _ = build_units(scopes)
        question_6 = next(unit for unit in units if unit["question_start"] == 6)
        question_7 = next(unit for unit in units if unit["question_start"] == 7)
        self.assertEqual(len(units), 4)
        self.assertEqual(question_7["structure_confidence"], "low")
        self.assertNotIn("일곱 번째", question_6["question_text"])
        self.assertIn("일곱 번째", question_7["question_text"])
        self.assertNotIn("첫 문제", question_7["question_text"])
        self.assertNotIn("마지막 문제", question_7["question_text"])

    def test_missing_anchor_preserves_scope_when_layout_is_ambiguous(self):
        records = [
            make_page(
                [
                    ("※ [5~8] 다음을 읽고 알맞은 것을 고르십시오.", 100),
                    ("5. 첫 문제", 150),
                    ("6. 둘째 문제", 250),
                    ("번호를 잃은 내용 하나", 350),
                    ("번호를 잃은 내용 둘", 450),
                    ("8. 마지막 문제", 550),
                ],
                extraction_method="ocr",
            )
        ]
        scopes = parse_exam(records)
        units, _ = build_units(scopes)
        question_7 = next(unit for unit in units if unit["question_start"] == 7)

        self.assertIn("첫 문제", question_7["question_text"])
        self.assertIn("마지막 문제", question_7["question_text"])

    def test_next_question_stem_does_not_leak_before_its_anchor(self):
        records = [
            make_page(
                [
                    ("※ [9~10] 다음을 읽고 알맞은 것을 고르십시오.", 100),
                    ("9. 아홉 번째 문제", 150),
                    ("아홉 번째 선택지", 190),
                    ("열 번째 문제의 앞선 줄", 240),
                    ("10.", 250),
                    ("열 번째 선택지", 290),
                ]
            )
        ]
        scopes = parse_exam(records)
        units, _ = build_units(scopes)
        question_9, question_10 = units

        self.assertNotIn("열 번째 문제의 앞선 줄", question_9["question_text"])
        self.assertIn("열 번째 문제의 앞선 줄", question_10["question_text"])

    def test_shared_context_excludes_sibling_specific_stem(self):
        records = [
            make_page(
                [
                    ("※ [19~20] 다음 글을 읽고 물음에 답하십시오.", 100),
                    ("두 문제에 필요한 공동 지문입니다.", 140),
                    ("19. 첫 번째 물음", 300),
                    ("첫 번째 선택지", 340),
                    ("두 번째 물음의 앞선 줄", 420),
                    ("20.", 430),
                    ("두 번째 선택지", 470),
                ]
            )
        ]
        scopes = parse_exam(records)
        units, _ = build_units(scopes)
        question_19, question_20 = units

        self.assertIn("공동 지문", question_19["text"])
        self.assertIn("공동 지문", question_20["text"])
        self.assertNotIn("두 번째 물음의 앞선 줄", question_19["question_text"])
        self.assertNotIn("첫 번째 물음", question_20["question_text"])
        self.assertIn("두 번째 물음의 앞선 줄", question_20["question_text"])

    def test_shared_context_infers_missing_first_sibling_from_clear_gap(self):
        records = [
            make_page(
                [
                    ("※ [19~20] 다음 글을 읽고 물음에 답하십시오.", 100),
                    ("두 문제에 필요한 공동 지문 첫 줄입니다.", 140),
                    ("공동 지문 둘째 줄입니다.", 165),
                    ("첫 번째 물음의 번호가 없습니다.", 260),
                    ("첫 번째 물음 선택지입니다.", 290),
                    ("20. 두 번째 물음", 430),
                    ("두 번째 물음 선택지입니다.", 460),
                ],
                extraction_method="ocr",
            )
        ]
        scopes = parse_exam(records)
        units, contexts = build_units(scopes)
        question_19, question_20 = units

        self.assertEqual(question_19["structure_confidence"], "low")
        self.assertIn("공동 지문", contexts[0]["text"])
        self.assertNotIn("첫 번째 물음", contexts[0]["text"])
        self.assertIn("첫 번째 물음", question_19["question_text"])
        self.assertNotIn("두 번째 물음", question_19["question_text"])
        self.assertIn("공동 지문", question_20["text"])
        self.assertNotIn("첫 번째 물음", question_20["question_text"])

    def test_shared_context_infers_missing_last_sibling_from_clear_gap(self):
        records = [
            make_page(
                [
                    ("※ [19~20] 다음 글을 읽고 물음에 답하십시오.", 100),
                    ("두 문제에 필요한 공동 지문입니다.", 140),
                    ("19. 첫 번째 물음", 260),
                    ("첫 번째 물음 선택지입니다.", 290),
                    ("두 번째 물음의 번호가 없습니다.", 400),
                    ("두 번째 물음 선택지입니다.", 430),
                ],
                extraction_method="ocr",
            )
        ]
        scopes = parse_exam(records)
        units, _ = build_units(scopes)
        question_19, question_20 = units

        self.assertNotIn("두 번째 물음", question_19["question_text"])
        self.assertEqual(question_20["structure_confidence"], "low")
        self.assertIn("두 번째 물음", question_20["question_text"])
        self.assertNotIn("첫 번째 물음", question_20["question_text"])
        self.assertIn("공동 지문", question_20["text"])

    def test_shared_edge_missing_anchor_stays_broad_when_gaps_are_ambiguous(self):
        records = [
            make_page(
                [
                    ("※ [19~20] 다음 글을 읽고 물음에 답하십시오.", 100),
                    ("공동 지문의 첫 단락입니다.", 140),
                    ("공동 지문의 둘째 단락입니다.", 240),
                    ("첫 번째 물음의 번호가 없습니다.", 340),
                    ("첫 번째 물음 선택지입니다.", 370),
                    ("20. 두 번째 물음", 500),
                ],
                extraction_method="ocr",
            )
        ]
        scopes = parse_exam(records)
        units, contexts = build_units(scopes)
        question_19 = units[0]

        self.assertEqual(question_19["structure_confidence"], "low")
        self.assertIn("첫 번째 물음", contexts[0]["text"])
        self.assertIn("두 번째 물음", question_19["question_text"])

    def test_inferred_visual_start_only_tightens_crop_metadata(self):
        unit_6 = {
            "question_start": 6,
            "preview_regions": [
                {"page_number": 1, "start_y": 280.0, "end_y": 480.0}
            ],
        }
        unit_7 = {
            "question_start": 7,
            "preview_regions": [
                {"page_number": 1, "start_y": 100.0, "end_y": 480.0}
            ],
        }
        inferred = {7: (1, 365.0)}

        self.assertEqual(
            refine_visual_preview_regions(unit_6, inferred),
            [{"page_number": 1, "start_y": 280.0, "end_y": 365.0}],
        )
        self.assertEqual(
            refine_visual_preview_regions(unit_7, inferred),
            [{"page_number": 1, "start_y": 365.0, "end_y": 480.0}],
        )
        self.assertEqual(unit_7["preview_regions"][0]["start_y"], 100.0)

    def test_shared_context_bypasses_independent_visual_inference(self):
        self.assertEqual(
            infer_independent_visual_starts(
                {"structure_type": "shared_context"},
                doc=None,
            ),
            {},
        )

    def test_shared_edge_visual_inference_verifies_the_layout_gap(self):
        blocks = [
            {"order": 0, "page": 1, "bbox": [50, 140, 500, 155]},
            {"order": 1, "page": 1, "bbox": [50, 165, 500, 180]},
            {"order": 2, "page": 1, "bbox": [50, 260, 500, 275]},
            {"order": 3, "page": 1, "bbox": [50, 290, 500, 305]},
            {"order": 4, "page": 1, "bbox": [50, 430, 500, 445]},
        ]
        scope = {
            "structure_type": "shared_context",
            "question_numbers": [19, 20],
            "blocks": blocks,
            "shared_context_blocks": blocks[:4],
            "question_anchors": [
                {
                    "question_number": 20,
                    "block_order": 4,
                    "page": 1,
                    "bbox": [50, 430, 70, 445],
                }
            ],
        }
        doc = pymupdf.open()
        doc.new_page(width=595, height=842)
        try:
            inferred = infer_shared_visual_starts(scope, doc)
        finally:
            doc.close()

        self.assertEqual(set(inferred), {19})
        self.assertEqual(inferred[19][0], 1)
        self.assertGreater(inferred[19][1], 180.0)
        self.assertLess(inferred[19][1], 260.0)

    def test_shared_preview_separates_context_from_target_question(self):
        scope = {
            "structure_type": "shared_context",
            "question_numbers": [19, 20],
            "scope_start": {"page": 1, "y": 100.0},
            "scope_end": {"page": 1, "y": 700.0},
            "shared_context_blocks": [{"page": 1, "bbox": [50, 140, 500, 300]}],
            "question_anchors": [
                {"question_number": 19, "page": 1, "bbox": [50, 350, 70, 365]},
                {"question_number": 20, "page": 1, "bbox": [50, 500, 70, 515]},
            ],
        }
        unit = {
            "question_start": 20,
            "shared_context_id": "context-19-20",
            "preview_regions": [
                {"page_number": 1, "start_y": 100.0, "end_y": 700.0}
            ],
        }

        self.assertEqual(
            compose_shared_visual_preview_regions(scope, unit),
            [
                {
                    "page_number": 1,
                    "start_y": 100.0,
                    "end_y": 344.0,
                    "role": "shared_context",
                },
                {
                    "page_number": 1,
                    "start_y": 494.0,
                    "end_y": 700.0,
                    "role": "question",
                },
            ],
        )

    def test_shared_preview_uses_inferred_first_and_last_edge_starts(self):
        first_missing_scope = {
            "structure_type": "shared_context",
            "question_numbers": [19, 20],
            "scope_start": {"page": 1, "y": 100.0},
            "scope_end": {"page": 1, "y": 700.0},
            "shared_context_blocks": [{"page": 1, "bbox": [50, 140, 500, 300]}],
            "question_anchors": [
                {"question_number": 20, "page": 1, "bbox": [50, 500, 70, 515]},
            ],
        }
        question_20 = {
            "question_start": 20,
            "shared_context_id": "context-19-20",
            "preview_regions": [
                {"page_number": 1, "start_y": 100.0, "end_y": 700.0}
            ],
        }
        self.assertEqual(
            compose_shared_visual_preview_regions(
                first_missing_scope,
                question_20,
                {19: (1, 344.0)},
            ),
            [
                {
                    "page_number": 1,
                    "start_y": 100.0,
                    "end_y": 344.0,
                    "role": "shared_context",
                },
                {
                    "page_number": 1,
                    "start_y": 494.0,
                    "end_y": 700.0,
                    "role": "question",
                },
            ],
        )

        last_missing_scope = {
            **first_missing_scope,
            "question_anchors": [
                {"question_number": 19, "page": 1, "bbox": [50, 350, 70, 365]},
            ],
        }
        question_19 = {**question_20, "question_start": 19}
        self.assertEqual(
            compose_shared_visual_preview_regions(
                last_missing_scope,
                question_19,
                {20: (1, 494.0)},
            )[-1],
            {
                "page_number": 1,
                "start_y": 344.0,
                "end_y": 494.0,
                "role": "question",
            },
        )


class TopikLexicalSearchTests(TestCase):
    def setUp(self):
        self.retriever = TopikRetriever.__new__(TopikRetriever)
        self.retriever.units = [
            {
                "exam_number": 60,
                "section": "reading",
                "question_start": 1,
                "question_end": 1,
                "question_numbers": [1],
                "pages": [5],
                "source_file": "topik_60.pdf",
                "chunk_type": "question",
                "text": "좋아하는 반면에 가격이 비싸다. 반면에 품질은 좋다.",
            }
        ]
        self.retriever._normalized_units = [
            (unit, self.retriever._normalize_text(unit["text"]))
            for unit in self.retriever.units
        ]

    def test_lexical_search_ignores_spacing_and_notation_hyphen(self):
        result = self.retriever.lexical_search("-는 반면에")
        self.assertEqual(result[0]["match_count"], 1)

    def test_empty_grammar_form_returns_no_results(self):
        self.assertEqual(self.retriever.lexical_search("-"), [])


class TopikSourceTests(TestCase):
    def test_manifest_has_unique_enabled_sources(self):
        sources = load_sources(DEFAULT_MANIFEST)
        self.assertEqual(len(sources), 7)
        self.assertEqual(len({source.exam_number for source in sources}), len(sources))
        self.assertEqual(len({source.file for source in sources}), len(sources))
        source_64 = next(source for source in sources if source.exam_number == 64)
        self.assertEqual(source_64.reading_pages, (5, 25))

    def test_native_text_usability_rejects_scan_only_page(self):
        self.assertFalse(text_is_usable(""))
        self.assertTrue(text_is_usable("한국어 문장이 충분히 들어 있는 디지털 시험 페이지입니다. " * 3))

    def test_auto_accepts_structurally_usable_native_candidate(self):
        candidate = make_native_candidate(
            [
                "※ [1~2] 다음 글을 읽고 알맞은 것을 고르십시오.",
                "1. 첫 번째 한국어 문제와 충분한 설명입니다.",
                "2. 두 번째 한국어 문제와 충분한 설명입니다.",
            ]
        )

        assessment = assess_native_structure([candidate])
        self.assertTrue(assessment["accepted"])
        self.assertEqual(choose_auto_extraction([candidate]), "native")

    def test_auto_rejects_fragmented_native_structure(self):
        candidate = make_native_candidate(
            [
                "※ 다음 한국어 글을 읽고 알맞은 것을 고르십시오.",
                "[", "1", "~", "2", "]",
                "1. 첫 번째 한국어 문제와 충분한 설명입니다.",
                "2. 두 번째 한국어 문제와 충분한 설명입니다.",
            ]
        )

        assessment = assess_native_structure([candidate])
        self.assertFalse(assessment["accepted"])
        self.assertEqual(choose_auto_extraction([candidate]), "ocr")

    def test_auto_selects_ocr_for_image_only_candidate(self):
        candidate = make_native_candidate([])
        self.assertEqual(choose_auto_extraction([candidate]), "ocr")


@skipUnless(DATA_FILE.exists(), "local TOPIK retrieval data is not installed")
class LocalTopikDataTests(TestCase):
    def test_every_exam_has_exactly_fifty_single_question_units(self):
        with DATA_FILE.open(encoding="utf-8") as file:
            units = json.load(file)
        counts = Counter(unit["exam_number"] for unit in units)
        self.assertEqual(counts, {36: 50, 37: 50, 41: 50, 47: 50, 52: 50, 60: 50, 64: 50})
        self.assertTrue(
            all(
                unit["question_numbers"] == [unit["question_start"]]
                and unit["question_start"] == unit["question_end"]
                for unit in units
            )
        )

    def test_topik_64_questions_9_to_12_are_independent_units(self):
        with DATA_FILE.open(encoding="utf-8") as file:
            units = json.load(file)
        selected = [
            unit for unit in units
            if unit["exam_number"] == 64 and 9 <= unit["question_start"] <= 12
        ]
        self.assertEqual([unit["question_numbers"] for unit in selected], [[9], [10], [11], [12]])
        self.assertTrue(all(unit["shared_context_id"] is None for unit in selected))

    def test_all_shared_context_references_are_valid_and_non_orphaned(self):
        with DATA_FILE.open(encoding="utf-8") as file:
            units = json.load(file)
        with CONTEXTS_FILE.open(encoding="utf-8") as file:
            contexts = json.load(file)
        context_ids = {context["shared_context_id"] for context in contexts}
        referenced = {
            unit["shared_context_id"]
            for unit in units
            if unit["shared_context_id"] is not None
        }
        self.assertEqual(context_ids, referenced)
        q19, q20 = [
            unit for unit in units
            if unit["exam_number"] == 64 and unit["question_start"] in {19, 20}
        ]
        self.assertEqual(q19["shared_context_id"], q20["shared_context_id"])

    def test_every_question_has_crop_metadata_including_visual_questions(self):
        with CROP_METADATA_FILE.open(encoding="utf-8") as file:
            metadata = json.load(file)
        self.assertEqual(len(metadata), 350)
        for exam_number in (36, 37, 41, 47, 52, 60, 64):
            for question in range(1, 51):
                self.assertIn(f"{exam_number}:{question}:{question}", metadata)

    def test_topik_64_missing_anchor_crop_is_locally_bounded(self):
        with CROP_METADATA_FILE.open(encoding="utf-8") as file:
            metadata = json.load(file)
        q5 = metadata["64:5:5"]["crops"][0]["bbox"]
        q6 = metadata["64:6:6"]["crops"][0]["bbox"]
        q7 = metadata["64:7:7"]["crops"][0]["bbox"]
        q8 = metadata["64:8:8"]["crops"][0]["bbox"]

        self.assertGreater(q6[1], q5[1])
        self.assertLess(q6[3], q7[3])
        self.assertGreater(q7[1], q5[3])
        self.assertLess(q7[3], q8[3])

    def test_shared_context_previews_keep_the_same_stimulus_start(self):
        with CROP_METADATA_FILE.open(encoding="utf-8") as file:
            metadata = json.load(file)
        q19 = metadata["64:19:19"]
        q20 = metadata["64:20:20"]

        self.assertEqual(q19["shared_context_id"], q20["shared_context_id"])
        self.assertEqual(q19["crops"][0]["bbox"][1], q20["crops"][0]["bbox"][1])

    def test_all_shared_previews_exclude_sibling_question_anchors(self):
        with CROP_METADATA_FILE.open(encoding="utf-8") as file:
            metadata = json.load(file)
        with GROUPS_FILE.open(encoding="utf-8") as file:
            scopes = {
                (scope["exam_number"], scope["instruction_id"]): scope
                for scope in json.load(file)
            }

        for item in metadata.values():
            if item.get("shared_context_id") is None:
                continue
            scope = scopes[(item["exam_number"], item["instruction_id"])]
            context_crops = [
                crop for crop in item["crops"] if crop.get("role") == "shared_context"
            ]
            question_crops = [
                crop for crop in item["crops"] if crop.get("role") == "question"
            ]
            self.assertTrue(context_crops)
            self.assertTrue(question_crops)

            target = item["question_start"]
            for anchor in scope["question_anchors"]:
                inside_context = any(
                    crop["page_number"] == anchor["page"]
                    and crop["bbox"][1] <= anchor["bbox"][1] <= crop["bbox"][3]
                    for crop in context_crops
                )
                inside_question = any(
                    crop["page_number"] == anchor["page"]
                    and crop["bbox"][1] <= anchor["bbox"][1] <= crop["bbox"][3]
                    for crop in question_crops
                )
                self.assertFalse(inside_context)
                self.assertEqual(
                    inside_question,
                    anchor["question_number"] == target,
                )

    def test_known_form_occurs_in_distinct_retrieval_units(self):
        with DATA_FILE.open(encoding="utf-8") as file:
            units = [
                unit
                for unit in json.load(file)
                if unit.get("content_type") != "visual_or_sparse"
            ]
        retriever = TopikRetriever.__new__(TopikRetriever)
        retriever.units = units
        retriever._normalized_units = [
            (unit, retriever._normalize_text(unit["text"])) for unit in units
        ]

        results = retriever.lexical_search("-는 반면에")
        keys = {
            (item["exam_number"], item["question_start"], item["question_end"])
            for item in results
        }
        self.assertEqual(len(results), 2)
        self.assertEqual(len(keys), len(results))


metadata_items = list(topik_preview_service.metadata.values())
preview_pdf = (
    PDF_DIR / metadata_items[0]["source_file"] if metadata_items else Path("missing")
)


@skipUnless(metadata_items and preview_pdf.exists(), "local TOPIK PDFs are not installed")
class TopikPreviewTests(TestCase):
    def test_preview_is_rendered_lazily_and_cached(self):
        item = metadata_items[0]
        args = (
            item["exam_number"],
            item["question_start"],
            item["question_end"],
            0,
        )
        topik_preview_service.render.cache_clear()

        first = topik_preview_service.render(*args)
        second = topik_preview_service.render(*args)

        self.assertTrue(first.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIs(first, second)
        self.assertEqual(topik_preview_service.render.cache_info().hits, 1)

    def test_topik_64_visual_and_existing_text_previews_render(self):
        for question in (5, 6, 7, 8):
            image = topik_preview_service.render(64, question, question, 0)
            self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        image = topik_preview_service.render(60, 13, 13, 0)
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_visual_independent_and_shared_previews_render_for_every_exam(self):
        for exam_number in (36, 37, 41, 47, 52, 60, 64):
            for question in (5, 11, 19):
                image = topik_preview_service.render(
                    exam_number, question, question, 0
                )
                self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_every_q48_to_q50_shared_crop_index_renders(self):
        for exam_number in (36, 37, 41, 47, 52, 60, 64):
            for question in (48, 49, 50):
                item = topik_preview_service.metadata[
                    f"{exam_number}:{question}:{question}"
                ]
                self.assertGreaterEqual(len(item["crops"]), 2)
                for crop_index in range(len(item["crops"])):
                    image = topik_preview_service.render(
                        exam_number, question, question, crop_index
                    )
                    self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))


@skipUnless(CHROMA_DIR.exists(), "local TOPIK Chroma data is not installed")
class LocalTopikChromaTests(TestCase):
    def test_visual_questions_are_not_indexed(self):
        collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
            "topik_examples"
        )
        stored_ids = set(collection.get(include=[])["ids"])
        with DATA_FILE.open(encoding="utf-8") as file:
            units = json.load(file)
        visual_ids = {
            f"topik_{unit['exam_number']}_{unit['question_start']}_{unit['question_end']}"
            for unit in units
            if not unit["indexable"]
        }
        self.assertTrue(visual_ids.isdisjoint(stored_ids))
        self.assertEqual(len(stored_ids), sum(unit["indexable"] for unit in units))
