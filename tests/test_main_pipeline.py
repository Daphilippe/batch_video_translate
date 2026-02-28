"""Tests for VideoTranslationPipeline — orchestrator logic in main.py.

Covers: config loading errors, run() control flow, _get_file_count(),
_run_hybrid_pipeline(), and argument parsing.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from main import VideoTranslationPipeline


def _write_config(tmp_path, config_dict):
    """Write a config dict as JSON and return its path."""
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(config_dict), encoding="utf-8")
    return str(config_path)


MINIMAL_CONFIG = {
    "translation": {
        "source_lang": "en",
        "target_lang": "fr",
        "cache_file": "data/cache.json",
        "max_chars_batch": 2000,
    },
}

FULL_CONFIG = {
    "whisper": {
        "bin_path": "C:/fake/whisper.exe",
        "model_path": "C:/fake/model.bin",
    },
    "llm_config": {
        "source_lang": "English",
        "target_lang": "French",
        "chunk_size": 10,
        "prompt_file": "configs/system_prompt_test.txt",
    },
    "translation": {
        "source_lang": "en",
        "target_lang": "fr",
        "cache_file": "data/cache.json",
        "max_chars_batch": 2000,
    },
}


# ── Config loading errors ────────────────────────────────────────────


class TestConfigLoading:
    """Pipeline config error handling."""

    def test_missing_config_file_raises(self, tmp_path):
        """Missing config file → ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Configuration file not found"):
            VideoTranslationPipeline(
                output_dir=str(tmp_path / "out"),
                config_path=str(tmp_path / "nonexistent.json"),
            )

    def test_invalid_json_raises(self, tmp_path):
        """Malformed JSON → ValueError with parse info."""
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("{{{INVALID", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            VideoTranslationPipeline(
                output_dir=str(tmp_path / "out"),
                config_path=str(bad_config),
            )


# ── _get_file_count ──────────────────────────────────────────────────


class TestGetFileCount:
    """Utility to count files or subdirectories."""

    def test_count_by_extension(self, tmp_path):
        """Counts files matching the given extension."""
        config_path = _write_config(tmp_path, MINIMAL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        folder = tmp_path / "files"
        folder.mkdir()
        (folder / "a.srt").write_text("x", encoding="utf-8")
        (folder / "b.srt").write_text("x", encoding="utf-8")
        (folder / "c.txt").write_text("x", encoding="utf-8")

        assert pipeline._get_file_count(folder, (".srt",)) == 2

    def test_count_dirs(self, tmp_path):
        """Counts subdirectories when extension is 'dir'."""
        config_path = _write_config(tmp_path, MINIMAL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        folder = tmp_path / "dirs"
        folder.mkdir()
        (folder / "sub1").mkdir()
        (folder / "sub2").mkdir()
        (folder / "file.txt").write_text("x", encoding="utf-8")

        assert pipeline._get_file_count(folder, "dir") == 2

    def test_nonexistent_returns_zero(self, tmp_path):
        """Non-existent path returns 0."""
        config_path = _write_config(tmp_path, MINIMAL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        assert pipeline._get_file_count(tmp_path / "ghost", (".srt",)) == 0


# ── run() control flow ───────────────────────────────────────────────


class TestRunControlFlow:
    """Pipeline run() dispatching and error handling."""

    def test_run_nonexistent_input_dir(self, tmp_path):
        """run() returns early when input dir doesn't exist."""
        config_path = _write_config(tmp_path, MINIMAL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Should not raise — logs error and returns
        pipeline.run(str(tmp_path / "missing"), mode="translate", engine="legacy")

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_run_translate_mode_legacy(self, mock_gt_class, _sleep, tmp_path):
        """run(mode='translate', engine='legacy') invokes LegacyTranslator."""
        mock_instance = MagicMock()
        mock_gt_class.return_value = mock_instance
        mock_instance.translate.return_value = "Bonjour"

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Create input dir (run() checks existence)
        (tmp_path / "in").mkdir()

        # Create a source SRT in clean_srt dir
        (pipeline.dirs["clean_srt"] / "test.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        pipeline.run(str(tmp_path / "in"), mode="translate", engine="legacy")

        # File should exist in intermediate dir (legacy_mt)
        assert (pipeline.dirs["legacy_mt"] / "test.srt").exists()

    def test_run_translate_mode_llm_local(self, tmp_path):
        """run(mode='translate', engine='llm-local') invokes LLMTranslator."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Create input dir (run() checks existence)
        (tmp_path / "in").mkdir()

        # Create a source SRT
        (pipeline.dirs["clean_srt"] / "test.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        # Patch the provider to avoid network calls
        with patch("main.LlamaCPPProvider") as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.name = "MockLLM"
            mock_provider.ask.return_value = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
            mock_provider_cls.return_value = mock_provider

            pipeline.run(str(tmp_path / "in"), mode="translate", engine="llm-local")

        assert (pipeline.dirs["llm_mt"] / "test.srt").exists()

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    def test_run_optimize_mode(self, mock_gt, _sleep, tmp_path):
        """run(mode='optimize') invokes SRTOptimizer."""
        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Create input dir (run() checks existence)
        (tmp_path / "in").mkdir()

        # Place raw SRT
        (pipeline.dirs["raw_srt"] / "test.srt").write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:02,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        pipeline.run(str(tmp_path / "in"), mode="optimize", engine="legacy")
        assert (pipeline.dirs["clean_srt"] / "test.srt").exists()


# ── _run_hybrid_pipeline ─────────────────────────────────────────────


class TestRunHybridPipeline:
    """Hybrid pipeline sub-step orchestration."""

    @patch("modules.legacy_translator.time.sleep")
    @patch("modules.legacy_translator.GoogleTranslator")
    @patch("main.LlamaCPPProvider")
    def test_hybrid_runs_all_sub_steps(self, mock_llama_cls, mock_gt_cls, _sleep, tmp_path):
        """_run_hybrid_pipeline generates L1, Mt, then refines."""
        mock_gt_instance = MagicMock()
        mock_gt_cls.return_value = mock_gt_instance
        mock_gt_instance.translate.return_value = "Bonjour"

        mock_llama = MagicMock()
        mock_llama.name = "MockLLM"
        mock_llama.ask.return_value = "1\n00:00:01,000 --> 00:00:03,000\nBonjour\n"
        mock_llama_cls.return_value = mock_llama

        config_path = _write_config(tmp_path, FULL_CONFIG)
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        # Place S1
        (pipeline.dirs["clean_srt"] / "test.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello\n",
            encoding="utf-8",
        )

        pipeline._run_hybrid_pipeline()

        # All three intermediate dirs should have been touched
        assert (pipeline.dirs["legacy_mt"] / "test.srt").exists()
        assert (pipeline.dirs["llm_mt"] / "test.srt").exists()


# ── _validate_config_section ─────────────────────────────────────────


class TestValidateConfigSection:
    """Granular config section validation."""

    def test_missing_section_raises(self, tmp_path):
        """Missing top-level section → ValueError."""
        config_path = _write_config(tmp_path, {})
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="Missing required config section"):
            pipeline._validate_config_section("whisper", ["bin_path"])

    def test_missing_key_raises(self, tmp_path):
        """Present section but missing key → ValueError."""
        config_path = _write_config(tmp_path, {"whisper": {"bin_path": "x"}})
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        with pytest.raises(ValueError, match="Missing required config key"):
            pipeline._validate_config_section("whisper", ["bin_path", "model_path"])

    def test_valid_section_passes(self, tmp_path):
        """Complete section → no error."""
        config_path = _write_config(tmp_path, {"whisper": {"bin_path": "x", "model_path": "y"}})
        pipeline = VideoTranslationPipeline(output_dir=str(tmp_path / "out"), config_path=config_path)

        pipeline._validate_config_section("whisper", ["bin_path", "model_path"])
