import json
import os
import tempfile
import unittest

import build_index


def meta(**overrides):
    value = {
        "benchmark": "skateboard",
        "prompt": "minimal",
        "model": "qwen38-flash-next",
        "endpoint": "http://localhost:8080/v1",
        "timestamp": "2026-09-27T12:00:00+02:00",
        "params": {"max_tokens": 128000, "reasoning_effort": "low"},
        "contributor": "asoldano",
        "hardware": {"gpu": "test"},
        "engine": {"name": "llama.cpp", "version": "1"},
        "quantization": "FP4",
    }
    value.update(overrides)
    return value


class ReasoningNormalizationTests(unittest.TestCase):
    def test_new_reasoning_is_preserved(self):
        value = meta(reasoning={
            "requested": "default",
            "effort": None,
            "provider_default": {"effort": "medium"},
        })
        self.assertEqual(build_index.normalize_reasoning(value)["requested"], "default")
        self.assertEqual(
            build_index.normalize_reasoning(value)["provider_default"]["effort"], "medium")

    def test_legacy_effort_is_normalized(self):
        self.assertEqual(
            build_index.normalize_reasoning(meta()),
            {"requested": "effort", "effort": "low"},
        )

    def test_legacy_no_reasoning_is_normalized(self):
        value = meta(params={"max_tokens": 128000, "no_reasoning": True})
        self.assertEqual(
            build_index.normalize_reasoning(value),
            {"requested": "disabled", "effort": None},
        )

    def test_missing_reasoning_is_unknown(self):
        value = meta(params={"max_tokens": 128000})
        self.assertEqual(build_index.normalize_reasoning(value)["requested"], "unknown")

    def test_invalid_new_reasoning_is_reported(self):
        value = meta(reasoning={"requested": "effort", "effort": "banana"})
        self.assertTrue(build_index.validate_reasoning(value, "bad"))

    def test_effort_is_rejected_for_default_mode(self):
        value = meta(reasoning={"requested": "default", "effort": "medium"})
        self.assertTrue(build_index.validate_reasoning(value, "bad"))

    def test_effort_member_is_required(self):
        value = meta(reasoning={"requested": "default"})
        self.assertTrue(build_index.validate_reasoning(value, "bad"))


class GroupIdentityTests(unittest.TestCase):
    def test_group_id_is_stable(self):
        self.assertEqual(
            build_index.group_id(meta()),
            "qwen38-flash-next-minimal-090426f4f5b7",
        )

    def test_effort_and_max_tokens_do_not_split_qwen_group(self):
        low = meta(params={"max_tokens": 128000, "reasoning_effort": "low"})
        xhigh = meta(params={"max_tokens": 160000, "reasoning_effort": "xhigh"})
        disabled = meta(params={"max_tokens": 128000, "no_reasoning": True})
        self.assertEqual(build_index.group_id(low), build_index.group_id(xhigh))
        self.assertEqual(build_index.group_id(low), build_index.group_id(disabled))

    def test_non_reasoning_configuration_splits_groups(self):
        original = meta()
        changes = {
            "endpoint": "http://localhost:8081/v1",
            "contributor": "someone-else",
            "hardware": {"gpu": "different"},
            "engine": {"name": "llama.cpp", "version": "2"},
            "quantization": "Q4",
            "params": {"max_tokens": 128000, "reasoning_effort": "low", "temperature": 0.5},
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                self.assertNotEqual(build_index.group_id(original), build_index.group_id(meta(**{key: value})))

    def test_builder_emits_flat_version_two_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "low.json"), "w") as file:
                json.dump(meta(), file)
            with open(os.path.join(directory, "xhigh.json"), "w") as file:
                json.dump(meta(params={"max_tokens": 160000, "reasoning_effort": "xhigh"}), file)
            output = os.path.join(directory, "index.json")
            entries, problems = build_index.build_index(directory, output)
            with open(output) as file:
                index = json.load(file)
            self.assertEqual(problems, [])
            self.assertEqual(index["version"], 2)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["group_id"], entries[1]["group_id"])

    def test_builder_reports_non_object_metadata_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "bad.json"), "w") as file:
                json.dump(["not", "an", "object"], file)
            with open(os.path.join(directory, "good.json"), "w") as file:
                json.dump(meta(), file)
            output = os.path.join(directory, "index.json")
            entries, problems = build_index.build_index(directory, output)
            self.assertEqual(len(entries), 1)
            self.assertEqual(len(problems), 1)
            self.assertIn("metadata must be an object", problems[0])

    def test_builder_reports_invalid_field_types_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            value = meta(endpoint=123, file=["not", "a", "path"])
            with open(os.path.join(directory, "bad-fields.json"), "w") as file:
                json.dump(value, file)
            output = os.path.join(directory, "index.json")
            entries, problems = build_index.build_index(directory, output)
            self.assertEqual(len(entries), 1)
            self.assertEqual(len(problems), 2)
            self.assertTrue(any("endpoint must be" in problem for problem in problems))
            self.assertTrue(any("file must be" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
