from types import SimpleNamespace

from streammuse.infrastructure.rap.audio_output import NullAudioSink
from streammuse.presentation.rap_demo.cli import RapAudioFactories, build_demo, build_parser


class QuietFactories(RapAudioFactories):
    def create_synthesizer(self):
        return SimpleNamespace()

    def create_phrase_warper(self, *, policy):
        return SimpleNamespace(policy=policy, rubberband_version="test")

    def create_lyric_stream(self, *, base_url):
        return SimpleNamespace(close=lambda: None,
                               get_capabilities=lambda: {"tempo_updates_v1", "bounded_turns_v1"})

    def create_bar_audio_source(self, *, base_url, opus_codec):
        return SimpleNamespace(close=lambda: None)

    def create_sink(self, *, output, audio_format, audio_file, audio_device):
        return NullAudioSink(audio_format=audio_format)


def test_v3_composes_inference_without_starting_audio_or_creating_another_session(tmp_path):
    args = build_parser().parse_args([
        "--rap-audio-renderer", "moss_aligned_remote", "--rap-protocol", "v3",
        "--audio-output", "wav", "--log-dir", str(tmp_path), "--tempo", "60",
    ])
    runtime = build_demo(args, audio_factories=QuietFactories())
    try:
        assert runtime.inference is not None
        snapshot = runtime.inference.snapshot()
        assert snapshot["session_id"] == runtime.publisher.session_id
        assert snapshot["config"]["template_id"] == runtime.controller.scenario.segments[0].template_id
        assert snapshot["state"] == "stopped"
        assert snapshot["audio"]["callback_count"] == 0
        assert snapshot["capabilities"]["renderer"] == "moss_lyric_stream"
        assert snapshot["capabilities"]["commit_quantum_ticks"] == 16
    finally:
        runtime.close()
