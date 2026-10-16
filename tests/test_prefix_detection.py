import torch

from fastdllm_generate import ForwardObservation
from omtp_observer import OMTPObserver
from prefix_detection import CommittedPrefixMatcher, audit_prefix_event
from test_omtp import CharacterTokenizer


def _fixture(names, text):
    tokenizer = CharacterTokenizer(''.join(names) + text + '\\"')
    return tokenizer, CommittedPrefixMatcher(tokenizer, names)


def test_shared_prefix_is_ambiguous():
    tokenizer, matcher = _fixture(['Audio-to-Audio', 'Audio-to-Text'], 'Audio-to')
    ids = tokenizer.encode('Audio-to') + [126] * 30
    assert matcher.match(ids, 0, 126) is None


def test_unique_prefix_triggers_without_future_tokens():
    tokenizer, matcher = _fixture(['Audio-to-Audio', 'weather'], 'Audio-to')
    ids = tokenizer.encode('Audio-to') + [126] * 30
    result = matcher.match(ids, 0, 126)
    assert result['predicted_tool'] == 'Audio-to-Audio'
    assert result['matching_candidate_count'] == 1
    assert not result['complete_name']


def test_diverging_committed_suffix_is_not_ignored():
    tokenizer, matcher = _fixture(['Object Detection'], 'Object Classification"')
    for text in ('Object Classification', 'Object Classification"'):
        ids = tokenizer.encode(text) + [126] * 30
        assert matcher.match(ids, 0, 126) is None


def test_committed_suffix_island_contradicts_candidate():
    tokenizer, matcher = _fixture(['weather'], 'weatXer')
    ids = tokenizer.encode('weatXer') + [126] * 5
    ids[2] = 126
    assert matcher.match(ids, 0, 126) is None


def test_complete_candidate_needs_a_real_name_boundary():
    tokenizer, matcher = _fixture(['Audio-to-Audio'], 'Audio-to-Audio Enhancement"')
    assert matcher.match(tokenizer.encode('Audio-to-Audio Enhancement"'), 0, 126) is None
    assert matcher.match(tokenizer.encode('Audio-to-Audio"'), 0, 126)['complete_name']


def test_prefix_does_not_start_at_an_interior_local_alignment():
    target = '[{"name":"Object Classification"}]'
    tokenizer = CharacterTokenizer(target + 'Object Detection')
    observer = OMTPObserver(tokenizer=tokenizer, candidate_names=['Object Detection'])
    ids = tokenizer.encode(target)
    start = target.index('Object')
    observer.baseline.steps.append({'generation_token_ids_after': ids})
    dummy = ForwardObservation(
        forward_index=1, block_index=0, block_step=0,
        elapsed_s=1., commit_elapsed_s=1.1, forward_time_s=.5,
        prompt_length=0, block_start=0, block_end=len(ids), mask_id=126,
        x=torch.tensor([ids]), committed_x=torch.tensor([ids]),
        logits=torch.zeros(1, len(ids), 127),
    )
    prefix, natural = observer._prefix_and_natural(dummy, start)
    assert prefix is None and natural is None


def test_model_invalid_name_after_valid_unique_prefix_is_separate():
    target = '[{"name":"Object Classification"}]'
    tokenizer, matcher = _fixture(['Object Detection'], target)
    start = target.index('Object')
    ids = tokenizer.encode(target)
    partial = ids[:start + len('Object')] + [126] * (len(ids) - start - len('Object'))
    evidence = matcher.match(partial, start, 126)
    event = {
        'source': 'prefix', 'predicted_tool': 'Object Detection',
        'predict_time_s': 1., 'prefix': 'Object', 'prefix_validation': evidence,
    }
    result = audit_prefix_event(
        event, generated_ids=ids, tokenizer=tokenizer,
        candidate_names=['Object Detection'],
    )
    assert result['status'] == 'model_invalid_name'
    assert result['counts_as_prefix_error'] is False
    assert event['predict_time_s'] == 1.


def test_unverified_or_unrelated_invalid_name_is_not_exempted():
    target = '[{"name":"Object Classification"},{"name":"weather"}]'
    tokenizer, matcher = _fixture(['Object Detection', 'weather'], target)
    event = {
        'source': 'prefix', 'predicted_tool': 'Object Detection',
        'predict_time_s': 1., 'prefix': 'Object',
    }
    kwargs = dict(generated_ids=tokenizer.encode(target), tokenizer=tokenizer,
                  candidate_names=['Object Detection', 'weather'])
    assert audit_prefix_event(event, **kwargs)['status'] == 'unresolved'
    event['prefix_validation'] = {
        'validated_online': True, 'generation_position': target.index('weather'),
        'prefix_literal': 'Object',
    }
    assert audit_prefix_event(event, **kwargs)['status'] == 'prediction_error'


def test_text_uniqueness_is_not_tokenization_uniqueness():
    class AlternativeBPE(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.startswith('Audio-to-Text'):
                return [125] + super().encode(text[8:])
            return super().encode(text)

    tokenizer = AlternativeBPE('Audio-to-Audio Audio-to-Text"')
    matcher = CommittedPrefixMatcher(tokenizer, ['Audio-to-Audio', 'Audio-to-Text'])
    ids = tokenizer.encode('Audio-to') + [126] * 30
    assert matcher.match(ids, 0, 126) is None
