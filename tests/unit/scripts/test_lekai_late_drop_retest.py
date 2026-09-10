import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import run_lekai_late_drop_retest as retest


def test_raw_comparison_keeps_empty_responses_and_ignores_wallclock_metadata(tmp_path):
    record = {'timestamp_request': 123, 'request_data': {'generation_start_tick': 4},
              'response_data': {'accompaniment': [], 'response_metadata': {
                  'part0_tokens': [169, 171], 'raw_tokens': [169, 170], 'structural_tokens': [],
              }}}
    path = tmp_path / 'inferences.json'
    path.write_text(json.dumps([record]))
    before = retest.signatures(tmp_path)
    assert len(before) == 1 and before[4]['events'] == []
    record['timestamp_request'] = 456
    path.write_text(json.dumps([record]))
    assert retest.signatures(tmp_path) == before
    record['response_data']['response_metadata']['part0_tokens'] = [1, 171]
    path.write_text(json.dumps([record]))
    assert retest.signatures(tmp_path) != before


def test_raw_comparison_rejects_duplicate_request_ticks(tmp_path):
    record = {'request_data': {'generation_start_tick': 4}, 'response_data': {
        'accompaniment': [], 'response_metadata': {
            'part0_tokens': [], 'raw_tokens': [], 'structural_tokens': [],
        }}}
    (tmp_path / 'inferences.json').write_text(json.dumps([record, record]))
    with pytest.raises(AssertionError):
        retest.signatures(tmp_path)


def test_condition_is_pure_lekai_not_single_prompt_model():
    assert retest.CONDITION == 'lekai_no_prompt'
    assert retest.formal.IDS[retest.CONDITION] == 'streammuse_v1_standard'
