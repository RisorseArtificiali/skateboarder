import json
import os
import tempfile
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest import mock

import bench


def args(**overrides):
    values = {
        "model": "example/model",
        "variant": "minimal",
        "base": "http://localhost:8080/v1",
        "max_tokens": 128000,
        "temperature": None,
        "reasoning_effort": None,
        "no_reasoning": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def meta(**overrides):
    value = {
        "benchmark": "skateboard",
        "prompt": "minimal",
        "model": "example/model",
        "endpoint": "http://localhost:8080/v1",
        "params": {"max_tokens": 128000, "reasoning_effort": "high"},
        "reasoning": {"requested": "effort", "effort": "high"},
        "contributor": "tester",
        "engine": {"name": "llama.cpp", "version": "1"},
    }
    value.update(overrides)
    return value


class RequestBodyTests(unittest.TestCase):
    def test_openrouter_default_omits_reasoning(self):
        body = bench.build_request_body(args(base="https://openrouter.ai/api/v1"))
        self.assertNotIn("reasoning", body)
        self.assertNotIn("reasoning_effort", body)

    def test_openrouter_effort_uses_unified_shape(self):
        body = bench.build_request_body(args(
            base="https://openrouter.ai/api/v1", reasoning_effort="xhigh"))
        self.assertEqual(body["reasoning"], {"effort": "xhigh"})
        self.assertNotIn("reasoning_effort", body)

    def test_openrouter_disabled_uses_none_effort(self):
        body = bench.build_request_body(args(
            base="https://openrouter.ai/api/v1", no_reasoning=True))
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertNotIn("chat_template_kwargs", body)

    def test_local_effort_is_passed_to_llama_cpp(self):
        body = bench.build_request_body(args(reasoning_effort="medium"))
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertNotIn("reasoning", body)

    def test_local_disabled_overrides_template_thinking(self):
        body = bench.build_request_body(args(no_reasoning=True))
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], False)

    def test_zero_temperature_is_sent(self):
        body = bench.build_request_body(args(temperature=0.0))
        self.assertEqual(body["temperature"], 0.0)

    def test_reasoning_flags_are_mutually_exclusive(self):
        parser = bench.build_parser()
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--model", "test", "--reasoning-effort", "high", "--no-reasoning"])


class OpenRouterCatalogTests(unittest.TestCase):
    def test_catalog_reasoning_is_snapshotted(self):
        payload = {"data": [{
            "id": "example/model",
            "reasoning": {
                "default_enabled": True,
                "default_effort": "medium",
                "mandatory": False,
                "supported_efforts": ["high", "medium", "low"],
            },
        }]}
        with mock.patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(payload).encode())):
            snapshot = bench.fetch_openrouter_reasoning(
                "https://openrouter.ai/api/v1", "example/model", {})
        self.assertEqual(snapshot["effort"], "medium")
        self.assertEqual(snapshot["enabled"], True)
        self.assertEqual(snapshot["source"], "openrouter-models")

    def test_malformed_catalog_is_non_fatal(self):
        payload = {"data": ["not-a-model"]}
        with mock.patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(payload).encode())):
            with mock.patch("sys.stderr"):
                snapshot = bench.fetch_openrouter_reasoning(
                    "https://openrouter.ai/api/v1", "example/model", {})
        self.assertIsNone(snapshot)


class OutputNameTests(unittest.TestCase):
    def test_preferred_names_include_explicit_reasoning(self):
        self.assertEqual(
            bench.preferred_output_name(args(reasoning_effort="high")),
            "example-model-minimal-reasoning_high",
        )
        self.assertEqual(
            bench.preferred_output_name(args(no_reasoning=True)),
            "example-model-minimal-no_reasoning",
        )
        self.assertEqual(bench.preferred_output_name(args()), "example-model-minimal")

    def test_same_request_identity_overwrites_preferred_name(self):
        run_args = args(reasoning_effort="high")
        current = meta()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "example-model-minimal-reasoning_high.json")
            with open(path, "w") as file:
                json.dump(current, file)
            self.assertEqual(
                bench.output_name(run_args, current, directory),
                "example-model-minimal-reasoning_high",
            )

    def test_different_configuration_gets_stable_hash(self):
        run_args = args(reasoning_effort="high")
        current = meta(params={"max_tokens": 160000, "reasoning_effort": "high"})
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "example-model-minimal-reasoning_high.json")
            with open(path, "w") as file:
                json.dump(meta(), file)
            first = bench.output_name(run_args, current, directory)
            second = bench.output_name(run_args, current, directory)
            self.assertEqual(first, second)
            self.assertRegex(first, r"^example-model-minimal-reasoning_high-[0-9a-f]{10}$")

    def test_orphan_svg_is_not_overwritten(self):
        run_args = args(reasoning_effort="high")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "example-model-minimal-reasoning_high.svg")
            with open(path, "w") as file:
                file.write("intentionally unrelated")
            name = bench.output_name(run_args, meta(), directory)
            self.assertRegex(name, r"^example-model-minimal-reasoning_high-[0-9a-f]{10}$")

    def test_malformed_existing_sidecar_is_not_overwritten(self):
        run_args = args(reasoning_effort="high")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "example-model-minimal-reasoning_high.json")
            with open(path, "w") as file:
                json.dump(["not", "an", "object"], file)
            name = bench.output_name(run_args, meta(), directory)
            self.assertRegex(name, r"^example-model-minimal-reasoning_high-[0-9a-f]{10}$")

    def test_occupied_short_hash_uses_a_longer_hash(self):
        run_args = args(reasoning_effort="high")
        current = meta(params={"max_tokens": 160000, "reasoning_effort": "high"})
        with tempfile.TemporaryDirectory() as directory:
            preferred = os.path.join(directory, "example-model-minimal-reasoning_high.json")
            with open(preferred, "w") as file:
                json.dump(meta(), file)
            short_name = bench.output_name(run_args, current, directory)
            with open(os.path.join(directory, f"{short_name}.json"), "w") as file:
                json.dump(meta(model="different/model"), file)
            longer_name = bench.output_name(run_args, current, directory)
            self.assertRegex(longer_name, r"^example-model-minimal-reasoning_high-[0-9a-f]{16}$")

    def test_failed_rerun_removes_only_its_stale_svg(self):
        run_args = args(reasoning_effort="high")
        current = meta(error="request failed")
        with tempfile.TemporaryDirectory() as directory:
            old_results = bench.RESULTS_DIR
            bench.RESULTS_DIR = directory
            try:
                name = bench.preferred_output_name(run_args)
                with open(os.path.join(directory, f"{name}.json"), "w") as file:
                    json.dump(current, file)
                svg_path = os.path.join(directory, f"{name}.svg")
                with open(svg_path, "w") as file:
                    file.write("stale")
                with mock.patch("builtins.print"):
                    bench.write_meta(run_args, current, None)
                self.assertFalse(os.path.exists(svg_path))
            finally:
                bench.RESULTS_DIR = old_results

    def test_failed_run_keeps_raw_response_for_inspection(self):
        run_args = args(reasoning_effort="high")
        current = meta(error="no SVG in response (finish_reason='error', content_len=18)")
        with tempfile.TemporaryDirectory() as directory:
            old_results = bench.RESULTS_DIR
            bench.RESULTS_DIR = directory
            try:
                name = bench.preferred_output_name(run_args)
                with mock.patch("builtins.print"):
                    bench.write_meta(run_args, current, None, raw="<svg> never closed")
                raw_path = os.path.join(directory, f"{name}.txt")
                with open(raw_path) as file:
                    self.assertEqual(file.read(), "<svg> never closed")
                self.assertEqual(current["raw_output"], f"results/{name}.txt")
                with open(os.path.join(directory, f"{name}.json")) as file:
                    self.assertEqual(json.load(file)["raw_output"], f"results/{name}.txt")
            finally:
                bench.RESULTS_DIR = old_results

    def test_successful_rerun_removes_stale_raw_response(self):
        run_args = args(reasoning_effort="high")
        current = meta()
        with tempfile.TemporaryDirectory() as directory:
            old_results = bench.RESULTS_DIR
            bench.RESULTS_DIR = directory
            try:
                name = bench.preferred_output_name(run_args)
                raw_path = os.path.join(directory, f"{name}.txt")
                with open(raw_path, "w") as file:
                    file.write("stale dump of a previous failed run")
                with mock.patch("builtins.print"):
                    bench.write_meta(run_args, current, "<svg>ok</svg>", raw="ignored on success")
                self.assertFalse(os.path.exists(raw_path))
                self.assertNotIn("raw_output", current)
                with open(os.path.join(directory, f"{name}.svg")) as file:
                    self.assertEqual(file.read(), "<svg>ok</svg>")
            finally:
                bench.RESULTS_DIR = old_results


if __name__ == "__main__":
    unittest.main()
