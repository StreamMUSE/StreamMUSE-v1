import numpy as np

from streammuse.infrastructure.inference.lekai_model.my_tokenizer import PianoRollTokenizer


def _build_tokenizer() -> PianoRollTokenizer:
    return PianoRollTokenizer(
        patch_h=1,
        patch_w=4,
        marker_offset=81,
        measures_length=88,
        end_marker_part0=170,
        end_marker_part1=171,
        empty_marker=169,
        img_h=88,
    )


def test_encode_decode_roundtrip_empty_pianoroll():
    tokenizer = _build_tokenizer()
    pianoroll = np.zeros((2, 88, 8), dtype=np.float32)

    encoded = tokenizer.encode(pianoroll)
    decoded = tokenizer.decode(encoded)

    assert decoded.shape == pianoroll.shape
    assert np.array_equal(decoded, pianoroll)


def test_encode_decode_roundtrip_with_notes():
    tokenizer = _build_tokenizer()
    pianoroll = np.zeros((2, 88, 8), dtype=np.float32)
    pianoroll[0, 60, 0:4] = 1.0
    pianoroll[1, 60, 0] = 1.0
    pianoroll[0, 64, 4:8] = 1.0
    pianoroll[1, 64, 4] = 1.0

    encoded = tokenizer.encode(pianoroll)
    decoded = tokenizer.decode(encoded)

    assert decoded.shape == pianoroll.shape
    assert np.array_equal(decoded, pianoroll)


def test_compress_decompress_roundtrip():
    tokenizer = _build_tokenizer()

    token_matrix = np.zeros((3, 88), dtype=np.int64)
    token_matrix[0, 0] = 5
    token_matrix[1, 3] = 7
    token_matrix[2, 87] = 9

    compressed = tokenizer.compress_tokens(token_matrix, end_marker=171)
    decompressed = tokenizer.decompress_tokens(compressed, end_marker_id=171)

    assert np.array_equal(decompressed, token_matrix)


def test_edge_case_all_ones_single_beat():
    tokenizer = _build_tokenizer()

    # Single beat = one patch in time for patch_w=4.
    pianoroll = np.ones((2, 88, 4), dtype=np.float32)

    encoded = tokenizer.encode(pianoroll)
    decoded = tokenizer.decode(encoded)

    assert decoded.shape == pianoroll.shape
    assert np.array_equal(decoded, pianoroll)


def test_edge_case_single_patch_sparse_note():
    tokenizer = _build_tokenizer()

    pianoroll = np.zeros((2, 88, 4), dtype=np.float32)
    pianoroll[0, 40, 2] = 1.0
    pianoroll[1, 40, 2] = 1.0

    encoded = tokenizer.encode(pianoroll)
    decoded = tokenizer.decode(encoded)

    assert decoded.shape == (2, 88, 4)
    assert np.array_equal(decoded, pianoroll)
