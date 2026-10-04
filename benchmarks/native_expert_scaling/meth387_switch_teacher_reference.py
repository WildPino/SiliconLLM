"""Official cached teacher-ID trajectory, raw logits before forcing processor."""
import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList


def official_teacher_cached(model, source, decoder_ids):
    assert not model.training and source.shape[0] == 1
    assert 0 < len(decoder_ids) <= 64 and decoder_ids[0] == 0
    assert all(0 <= value < model.config.vocab_size for value in decoder_ids)
    raw = []

    class CaptureTeacher(LogitsProcessor):
        def __call__(self, input_ids, scores):
            position = input_ids.shape[1] - 1
            assert 0 <= position < len(decoder_ids)
            assert input_ids[0].tolist() == decoder_ids[:position + 1]
            assert scores.shape == (1, model.config.vocab_size)
            values = scores.detach().cpu().numpy()[0].copy()
            assert np.isfinite(values).all()
            raw.append(values)
            next_id = decoder_ids[position + 1] if position + 1 < len(decoder_ids) else 0
            forced = torch.full_like(scores, -torch.inf)
            forced[:, next_id] = 0
            return forced

    with torch.no_grad():
        result = model.generate(input_ids=source, max_new_tokens=len(decoder_ids),
                                do_sample=False, num_beams=1, use_cache=True,
                                decoder_start_token_id=0, eos_token_id=None,
                                pad_token_id=0,
                                logits_processor=LogitsProcessorList([CaptureTeacher()]),
                                return_dict_in_generate=True, output_scores=False)
    assert result.sequences[0].tolist() == decoder_ids + [0]
    assert len(raw) == len(decoder_ids)
    return np.stack(raw)


def tiny_controls(B, Config, Model):
    """Qualify API capture/trajectory before any complete-source quality score."""
    results = []
    for capacity in (1, 64):
        torch.manual_seed(328)
        config = Config(d_model=8, d_ff=16, d_kv=4, num_heads=2, num_experts=2,
                        num_layers=2, num_decoder_layers=2, num_sparse_encoder_layers=1,
                        num_sparse_decoder_layers=1, expert_capacity=capacity,
                        dropout_rate=0., router_jitter_noise=0., vocab_size=32,
                        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
                        router_dtype='float32')
        model = Model(config).eval()
        source = torch.tensor([[2, 3, 4, 5, 6, 7]])
        decoder = [0, 8, 9, 10]
        reference = B.torch_reference(model, source, torch.tensor([decoder]))[2]
        actual = official_teacher_cached(model, source, decoder)
        wrong = official_teacher_cached(model, source, [0, 9, 9, 10])
        error = B.relative(actual, reference)
        wrong_error = B.relative(wrong, reference)
        top1 = bool(np.array_equal(actual.argmax(-1), reference.argmax(-1)))
        assert error <= 1e-6 and top1 and wrong_error > 1e-4
        results.append({'capacity': capacity, 'relative_logit_l2': error,
                        'teacher_top1_exact': top1,
                        'wrong_decoder_id_relative_l2': wrong_error,
                        'wrong_decoder_id_detected': True, 'passed': True})
    return results
