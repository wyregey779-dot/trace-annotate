"""Person detections retain model geometry and fail closed without a provider."""
import numpy as np
import pytest

from trace_annotate.extractors.persons import PersonExtractor, decode_person_outputs


def outputs():
    scores_all, boxes_all = [], []
    for stride in (8, 16, 32):
        count = (416 // stride) ** 2
        scores = np.zeros((1, count, 80), dtype=np.float32)
        boxes = np.full((1, count, 32), -30, dtype=np.float32)
        boxes.reshape(1, count, 4, 8)[:, :, :, 2] = 30
        scores_all.append(scores)
        boxes_all.append(boxes)
    return scores_all + boxes_all


def test_missing_model_never_reports_zero_people(tmp_path):
    extractor = PersonExtractor(str(tmp_path / 'missing.onnx'))
    result = extractor.detect_people(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result['status'] == 'not_run'
    assert result['person_count'] is None
    assert result['persons_detected'] == []


def test_invalid_model_is_failed_not_missing(tmp_path):
    path = tmp_path / '损坏.onnx'
    path.write_bytes(b'not an onnx model')
    extractor = PersonExtractor(str(path))
    result = extractor.detect_people(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result['status'] == 'failed'
    assert result['person_count'] is None


def test_person_class_and_letterbox_are_recovered():
    values = outputs()
    index = 20 * 52 + 20
    values[0][0, index, 0] = .9
    values[0][0, index + 4, 2] = .98  # car must not become a person
    result = decode_person_outputs(values, (832, 416), (.5, .5, 0., 104.))
    assert len(result) == 1
    assert result[0]['box'] == pytest.approx([295., 87., 359., 151.])
    assert result[0]['confidence'] == pytest.approx(.9)
    assert result[0]['coordinate_space'] == 'decoded_frame_xyxy'


def test_nms_rejects_duplicate_anchor_without_losing_second_person():
    values = outputs()
    values[0][0, 20 * 52 + 20, 0] = .9
    values[0][0, 20 * 52 + 21, 0] = .8
    values[0][0, 35 * 52 + 35, 0] = .85
    assert len(decode_person_outputs(values, (416, 416), (1., 1., 0., 0.))) == 2


@pytest.mark.parametrize('bad', ['count', 'shape', 'nan'])
def test_unexpected_model_output_is_rejected(bad):
    values = outputs()
    if bad == 'count':
        values.pop()
    elif bad == 'shape':
        values[0] = np.zeros((1, 3, 80))
    else:
        values[0][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        decode_person_outputs(values, (416, 416), (1., 1., 0., 0.))


def test_empty_success_is_distinct_from_no_provider():
    assert decode_person_outputs(outputs(), (416, 416), (1., 1., 0., 0.)) == []


def test_weight_pin_detects_changed_file_before_session_load(tmp_path):
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'changed')
    extractor = PersonExtractor(str(path), expected_sha256='0' * 64)
    assert extractor.detect_people(np.zeros((20, 20, 3), dtype=np.uint8))['reason'] == 'weights_hash_mismatch'


def test_model_byte_budget_is_checked_before_reading(tmp_path, monkeypatch):
    from trace_annotate.extractors import persons
    path = tmp_path / 'too-large.onnx'
    path.write_bytes(b'1234')
    monkeypatch.setattr(persons, 'MAX_MODEL_BYTES', 3, raising=False)
    def must_not_read(*args, **kwargs):
        raise AssertionError('oversized model was read')
    monkeypatch.setattr(type(path), 'read_bytes', must_not_read)
    result = PersonExtractor(str(path)).detect_people(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result['status'] == 'failed'
    assert result['reason'] == 'person_model_exceeds_byte_budget'


def test_custom_weights_do_not_inherit_official_model_name_or_license(tmp_path):
    path = tmp_path / 'custom.onnx'
    path.write_bytes(b'unknown local model')
    identity = PersonExtractor(str(path)).code_identity
    assert identity['model_name'] == 'unverified_local_onnx'
    assert identity['license'] == 'unverified'


def test_official_model_attribution_requires_the_verified_weight_digest():
    extractor = PersonExtractor()
    extractor.weights_sha256 = '4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186'
    assert extractor.code_identity['license'] == 'Apache-2.0 (OpenCV Zoo)'
