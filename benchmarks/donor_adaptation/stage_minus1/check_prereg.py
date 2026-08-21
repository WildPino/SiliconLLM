#!/usr/bin/env python3
"""check_prereg.py -- the mechanical guard on the stage -1 pre-registration.

WHY THIS EXISTS
    The stage -1 pre-registration failed Controller review three times. The decisive finding of
    review #3 was that the prose "claimed fixes the document did not contain" -- five false closure
    claims -- which the reviewer called "a corrupted audit trail".

    Prose can assert that a field is pinned. This script cannot. It reads prereg.yaml against the
    required-key schema embedded below and exits NON-ZERO listing every field that is either

        ABSENT   -- the key is not in the file at all, or
        UNPINNED -- the key's value is a string beginning with the literal "UNPINNED:"

    A closure claim that this script does not agree with is, by construction, false.

WHY THE SCHEMA IS EMBEDDED IN THIS FILE AND NOT DERIVED FROM prereg.yaml
    If the required-key list were derived from prereg.yaml at runtime, deleting a key would delete
    its own requirement and the guard would stay silent -- the exact failure mode it exists to catch.
    The schema is a literal in this file: the contract lives outside the artefact it checks.

USAGE
    python check_prereg.py [path/to/prereg.yaml]
    python check_prereg.py --json          machine-readable report on stdout

EXIT CODES
    0  every required field is present and pinned
    1  at least one required field is ABSENT or UNPINNED
    2  the file could not be read or parsed
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("FATAL: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    sys.exit(2)

UNPINNED_MARKER = "UNPINNED:"

# ---------------------------------------------------------------------------------------------
# REQUIRED KEYS -- the contract. Dotted paths into prereg.yaml. Every one must exist and must not
# carry an UNPINNED: value for the pre-registration to be complete.
# ---------------------------------------------------------------------------------------------
REQUIRED_KEYS = [
    'schema_version',
    'stage',
    'status',
    'generated_utc',
    'governing_standard',
    'prior_reviews',
    'donor.repo_id',
    'donor.hub_revision_sha',
    'donor.read_from',
    'donor.gated',
    'donor.accessible_unauthenticated',
    'donor.licence',
    'donor.declared_dtype',
    'donor.declared_dtype_read_from',
    'donor.param_count_total',
    'donor.param_count_source',
    'donor.tie_word_embeddings',
    'donor.tie_word_embeddings_read_from',
    'donor.tie_word_embeddings_consequence',
    'donor.num_hidden_layers',
    'donor.hidden_size',
    'donor.intermediate_size',
    'donor.vocab_size',
    'donor.bos_token_id',
    'donor.eos_token_id',
    'donor.tokenizer_adds_bos_by_default',
    'donor.tokenizer_adds_bos_read_from',
    'donor.donor_vs_target_mismatch_declared',
    'method.name',
    'method.arxiv_id',
    'method.venue_claim',
    'method.repo_url',
    'method.repo_commit_sha',
    'method.repo_commit_date',
    'method.repo_commit_message',
    'method.repo_licence',
    'method.repo_read_from',
    'method.gradient_free',
    'method.gradient_free_evidence',
    'method.paper_calibration_sample_count',
    'method.paper_calibration_seqlen',
    'method.paper_calibration_corpus',
    'method.paper_calibration_read_from',
    'method.paper_reported_cost',
    'method.reference_command.read_from',
    'method.reference_command.line',
    'method.method_pinned_hyperparameters.low_quant_method',
    'method.method_pinned_hyperparameters.ssr',
    'method.method_pinned_hyperparameters.blocksize',
    'method.method_pinned_hyperparameters.percdamp',
    'method.method_pinned_hyperparameters.num_p',
    'method.method_pinned_hyperparameters.salient_metric',
    'method.method_pinned_hyperparameters.order2_group',
    'method.method_pinned_hyperparameters.disable_gptq',
    'method.method_pinned_hyperparameters.seed',
    'method.method_pinned_hyperparameters.nsamples',
    'method.method_pinned_hyperparameters.calib_seqlen',
    'method.method_pinned_hyperparameters.ppl_seqlen',
    'method.method_pinned_hyperparameters.read_from',
    'method.method_declared_dependency_pins.torch',
    'method.method_declared_dependency_pins.transformers',
    'method.method_declared_dependency_pins.tokenizers',
    'method.method_declared_dependency_pins.accelerate',
    'method.method_declared_dependency_pins.datasets',
    'method.method_declared_dependency_pins.read_from',
    'method.method_declared_dependency_pins.conflict_with_this_environment',
    'method.cpu_only_feasibility_smoke_test.status',
    'method.cpu_only_feasibility_smoke_test.protocol',
    'eval_slice.builder_proposal',
    'eval_slice.rationale',
    'eval_slice.corpus_id',
    'eval_slice.corpus_config',
    'eval_slice.corpus_split',
    'eval_slice.corpus_repo_type',
    'eval_slice.corpus_revision_sha',
    'eval_slice.corpus_revision_read_from',
    'eval_slice.corpus_file',
    'eval_slice.corpus_licence',
    'eval_slice.join_rule',
    'eval_slice.join_rule_read_from',
    'eval_slice.encoding',
    'eval_slice.sha256_joined_text',
    'eval_slice.sha256_evaluated_bytes',
    'eval_slice.joined_text_bytes_total',
    'eval_slice.byte_range_evaluated',
    'eval_slice.n_tokens_total',
    'eval_slice.n_windows',
    'eval_slice.n_tokens_evaluated',
    'eval_slice.n_target_tokens',
    'eval_slice.bytes_of_target_region',
    'eval_slice.bos_handling_verified',
    'eval_slice.tokenizer_repo_id',
    'eval_slice.tokenizer_revision_sha',
    'eval_slice.tokenizer_use_fast_in_method',
    'eval_slice.tokenizer_fast_slow_parity_check',
    'eval_slice.tokenizer_fast_slow_parity_result',
    'eval_slice.context_length',
    'eval_slice.stride',
    'eval_slice.stride_note',
    'eval_slice.tail_handling',
    'eval_slice.bos_handling',
    'eval_slice.first_token_of_window',
    'eval_slice.loss_reduction',
    'eval_slice.bpb_reduction',
    'eval_slice.bpb_vs_ppl_note',
    'eval_slice.decontamination_procedure',
    'eval_slice.decontamination_threshold',
    'eval_slice.decontamination_result_vs_calibration',
    'eval_slice.decontamination_result_vs_p62_codeval',
    'measured_prerequisites.rule',
    'measured_prerequisites.B_bytes_per_token.definition',
    'measured_prerequisites.B_bytes_per_token.procedure',
    'measured_prerequisites.B_bytes_per_token.value',
    'measured_prerequisites.B_secondary_whole_stream.definition',
    'measured_prerequisites.B_secondary_whole_stream.value',
    'measured_prerequisites.BPB_donor.definition',
    'measured_prerequisites.BPB_donor.procedure',
    'measured_prerequisites.BPB_donor.value',
    'measured_prerequisites.PPL_donor.definition',
    'measured_prerequisites.PPL_donor.value',
    'measured_prerequisites.B_per_anchor_model.definition',
    'measured_prerequisites.B_per_anchor_model.value',
    'bit_width_sweep.points',
    'bit_width_sweep.reference_point_is_not_a_curve_point',
    'bit_width_sweep.one_variable_claim_withdrawn',
    'bit_width_sweep.BLOCKING_FINDING_read_from_implementation',
    'bit_width_sweep.per_point_config.bf16_reference.method',
    'bit_width_sweep.per_point_config.bf16_reference.quantizer_class',
    'bit_width_sweep.per_point_config.bf16_reference.group_size',
    'bit_width_sweep.per_point_config.bf16_reference.outlier_policy',
    'bit_width_sweep.per_point_config.bf16_reference.effective_bits_per_weight',
    'bit_width_sweep.per_point_config.bf16_reference.scale_dtype',
    'bit_width_sweep.per_point_config.bf16_reference.rng_seed',
    'bit_width_sweep.per_point_config.bf16_reference.read_from',
    'bit_width_sweep.per_point_config.ternary_1p58.method',
    'bit_width_sweep.per_point_config.ternary_1p58.quantizer_class',
    'bit_width_sweep.per_point_config.ternary_1p58.init_threshold_rule',
    'bit_width_sweep.per_point_config.ternary_1p58.refinement',
    'bit_width_sweep.per_point_config.ternary_1p58.group_size',
    'bit_width_sweep.per_point_config.ternary_1p58.group_size_read_from',
    'bit_width_sweep.per_point_config.ternary_1p58.symmetric_or_affine',
    'bit_width_sweep.per_point_config.ternary_1p58.outlier_policy',
    'bit_width_sweep.per_point_config.ternary_1p58.clipping_procedure',
    'bit_width_sweep.per_point_config.ternary_1p58.internal_working_precision',
    'bit_width_sweep.per_point_config.ternary_1p58.fitted_parameters_per_layer',
    'bit_width_sweep.per_point_config.ternary_1p58.rng_seed',
    'bit_width_sweep.per_point_config.ternary_1p58.rng_seed_note',
    'bit_width_sweep.per_point_config.ternary_1p58.effective_bits_per_weight_formula',
    'bit_width_sweep.per_point_config.ternary_1p58.effective_bits_per_weight_at_fp16_scales',
    'bit_width_sweep.per_point_config.ternary_1p58.effective_bits_per_weight_at_fp32_scales',
    'bit_width_sweep.per_point_config.ternary_1p58.scale_dtype',
    'bit_width_sweep.per_point_config.4bit.method',
    'bit_width_sweep.per_point_config.4bit.quantizer_class',
    'bit_width_sweep.per_point_config.4bit.group_size',
    'bit_width_sweep.per_point_config.4bit.outlier_policy',
    'bit_width_sweep.per_point_config.4bit.effective_bits_per_weight',
    'bit_width_sweep.per_point_config.4bit.scale_dtype',
    'bit_width_sweep.per_point_config.4bit.rng_seed',
    'bit_width_sweep.per_point_config.3bit.method',
    'bit_width_sweep.per_point_config.3bit.quantizer_class',
    'bit_width_sweep.per_point_config.3bit.group_size',
    'bit_width_sweep.per_point_config.3bit.outlier_policy',
    'bit_width_sweep.per_point_config.3bit.effective_bits_per_weight',
    'bit_width_sweep.per_point_config.3bit.scale_dtype',
    'bit_width_sweep.per_point_config.3bit.rng_seed',
    'bit_width_sweep.per_point_config.2bit.method',
    'bit_width_sweep.per_point_config.2bit.quantizer_class',
    'bit_width_sweep.per_point_config.2bit.group_size',
    'bit_width_sweep.per_point_config.2bit.outlier_policy',
    'bit_width_sweep.per_point_config.2bit.effective_bits_per_weight',
    'bit_width_sweep.per_point_config.2bit.scale_dtype',
    'bit_width_sweep.per_point_config.2bit.rng_seed',
    'bit_width_sweep.x_axis',
    'bit_width_sweep.deliverable',
    'replicates.n',
    'replicates.axis_varied',
    'replicates.axis_seeds',
    'replicates.seed_0_is_the_paper_configuration',
    'replicates.second_axis_ptq_rng',
    'replicates.t_multiplier',
    'replicates.t_multiplier_basis',
    'replicates.SE_definition',
    'replicates.band_half_width',
    'replicates.paired_bootstrap',
    'replicates.SE_floor',
    'replicates.decision_reading',
    'gate_policy.donor_threshold',
    'gate_policy.where_the_stopping_rule_lives',
    'gate_policy.no_frozen_constant_in_B',
    'gate_policy.scope_PASS',
    'gate_policy.scope_FAIL',
    'anchors.source',
    'anchors.conversion_formula',
    'anchors.conversion_formula_note',
    'anchors.B_used_for_illustrative_column',
    'anchors.illustrative_column_is_not_a_decision_input',
    'anchors.rows',
    'anchors.frontier_is_a_range',
    'anchors.measured_B_qwen25_tokenizer',
    'anchors.measured_B_llama_sentencepiece',
    'anchors.measured_B_read_from',
    'anchors.tokenizer_B_asymmetry_finding',
    'anchors.frontier_range_at_measured_B',
    'anchors.donor_expectation_at_measured_B',
    'anchors.most_relevant_row',
    'anchors.expected_donor_direction',
    'anchors.independent_1to3B_datapoint',
    'anchors.anchor_dtype_asymmetry',
    'calibration.anchor_arm.corpus',
    'calibration.anchor_arm.n_samples',
    'calibration.anchor_arm.seqlen',
    'calibration.anchor_arm.sampling',
    'calibration.anchor_arm.why',
    'calibration.extended_arm.status',
    'calibration.extended_arm.n_samples',
    'calibration.extended_arm.seqlen',
    'calibration.extended_arm.composition',
    'calibration.extended_arm.sampling_order',
    'calibration.extended_arm.content_hash',
    'calibration.sensitivity_arm.protocol',
    'calibration.sensitivity_arm.note',
    'calibration.decontamination',
    'controls.control_1a.name',
    'controls.control_1a.what',
    'controls.control_1a.numeric_tolerance',
    'controls.control_1a.anchor_paper',
    'controls.control_1a.anchor_model',
    'controls.control_1a.tolerance_form',
    'controls.control_1a.on_miss',
    'controls.control_1b.name',
    'controls.control_1b.anchor_paper',
    'controls.control_1b.anchor_model',
    'controls.control_1b.anchor_model_substitution_note',
    'controls.control_1b.anchor_value_ppl_fp16',
    'controls.control_1b.anchor_value_ppl_w158',
    'controls.control_1b.numeric_tolerance',
    'controls.control_1b.tolerance_form',
    'controls.control_1b.on_miss',
    'controls.control_1b.feasibility',
    'controls.control_2.name',
    'controls.control_2.perturbation_family',
    'controls.control_2.eps_grid',
    'controls.control_2.rung_count',
    'controls.control_2.structures.redundant_rung',
    'controls.control_2.structures.non_redundant_rung',
    'controls.control_2.structures.second_non_redundant_rung',
    'controls.control_2.reported_quantity',
    'controls.control_2.numeric_tolerance',
    'controls.control_2.failure_to_fire_response',
    'controls.control_3.name',
    'controls.control_3.protocol',
    'controls.control_3.numeric_tolerance',
    'controls.control_3.both_directions_logged',
    'controls.control_3.known_limitation',
    'controls.control_4.name',
    'controls.control_4.protocol',
    'controls.control_4.exercised_both_directions',
    'controls.control_4.numeric_tolerance',
    'controls.control_4.architecture_numeric_probe',
    'controls.control_4.architecture_numeric_probe_value',
    'controls.control_4.tied_embeddings_assertion',
    'controls.control_5.name',
    'controls.control_5.protocol',
    'controls.control_5.numeric_tolerance',
    'controls.control_5.expected_direction',
    'controls.control_5.on_bit_identical',
    'controls.control_5.both_directions_logged',
    'controls.control_6.name',
    'controls.control_6.compared_quantity',
    'controls.control_6.cpu_leg_dtype',
    'controls.control_6.arms_covered',
    'controls.control_6.numeric_tolerance',
    'controls.control_6.tolerance_note',
    'controls.control_6.predicted_gap_sign_and_magnitude',
    'controls.control_7.name',
    'controls.control_7.protocol',
    'controls.control_7.numeric_tolerance',
    'controls.control_7.both_directions_logged',
    'controls.control_7.flags_read_back',
    'environment.target_device',
    'environment.target_device_uuid',
    'environment.target_device_uuid_read_from',
    'environment.excluded_device',
    'environment.CUDA_DEVICE_ORDER',
    'environment.CUDA_VISIBLE_DEVICES',
    'environment.CUDA_VISIBLE_DEVICES_note',
    'environment.assert_device_count_equals_1',
    'environment.assert_capability',
    'environment.assert_uuid_matches',
    'environment.assert_materialised_parameter_device',
    'environment.device_map_auto',
    'environment.weight_dtype',
    'environment.loss_accumulation_dtype',
    'environment.arms_share_dtype_policy',
    'environment.quantizer_internal_precision',
    'environment.precision_flags.cuda_matmul_allow_bf16_reduced_precision_reduction',
    'environment.precision_flags.cuda_matmul_allow_fp16_reduced_precision_reduction',
    'environment.precision_flags.cuda_matmul_allow_tf32',
    'environment.precision_flags.cudnn_allow_tf32',
    'environment.precision_flags.cudnn_benchmark',
    'environment.precision_flags.float32_matmul_precision',
    'environment.precision_flags.preferred_linalg_library',
    'environment.precision_flags.torch_attribute_map',
    'environment.precision_flags.preferred_linalg_library_note',
    'environment.precision_flags.attn_implementation',
    'environment.precision_flags.attn_implementation_note',
    'environment.precision_flags.tf32_status_note',
    'environment.flags_read_back_into_manifest',
    'environment.autocast',
    'environment.use_deterministic_algorithms',
    'environment.use_deterministic_algorithms_attr',
    'environment.use_deterministic_algorithms_warn_only',
    'environment.use_deterministic_algorithms_on_raise',
    'environment.CUBLAS_WORKSPACE_CONFIG',
    'environment.seeds.python_random',
    'environment.seeds.numpy',
    'environment.seeds.torch_manual_seed',
    'environment.seeds.torch_cuda_manual_seed_all',
    'environment.seeds.calibration_draw_seeds',
    'environment.harness_identity_recorded.torch_version',
    'environment.harness_identity_recorded.cuda_runtime_version',
    'environment.harness_identity_recorded.cudnn_version',
    'environment.harness_identity_recorded.driver_version',
    'environment.harness_identity_recorded.python_version',
    'environment.harness_identity_recorded.transformers_version',
    'environment.harness_identity_recorded.os',
    'environment.host_resources.ram_gb',
    'environment.host_resources.free_disk_gb',
    'feasibility.control_1b_vram.problem_as_stated',
    'feasibility.control_1b_vram.resolution',
    'feasibility.control_1b_vram.vram_arithmetic_llama_7b',
    'feasibility.control_1b_vram.verdict',
    'feasibility.control_1b_vram.cost',
    'feasibility.control_1b_vram.fallback_if_wrong',
    'feasibility.llama2_gate.checked',
    'feasibility.llama2_gate.method',
    'feasibility.llama2_gate.result_api',
    'feasibility.llama2_gate.result_file',
    'feasibility.llama2_gate.verdict',
    'feasibility.llama2_gate.consequence_if_unresolved',
    'feasibility.llama2_gate.resolution_adopted',
    'feasibility.llama2_gate.rejected_option',
    'install.status',
    'install.current_state',
    'install.recommended_command',
    'install.wheel_verified',
    'install.why_cu126_not_cu130',
    'install.why_same_version',
    'install.additional_packages_host',
    'install.additional_packages_method_venv',
    'install.post_install_verification',
    'on_failure.conclusion_not_drawn',
    'on_failure.question_asked',
    'on_failure.cost_note',
]

# Keys structurally allowed to remain UNPINNED forever WITHOUT blocking, because pinning them is
# impossible in principle rather than merely not done yet. THIS LIST IS DELIBERATELY EMPTY. It
# exists so that exempting a field requires a visible diff in code, never a sentence in prose.
STRUCTURAL_EXEMPTIONS = []


def resolve(tree, dotted):
    """Return (found, value). Walks a dotted path through nested dicts."""
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def leaf_paths(node, prefix=None):
    """Yield (dotted_path, value) for every leaf. A list of dicts counts as one leaf."""
    prefix = prefix or []
    if isinstance(node, dict):
        for key, value in node.items():
            for item in leaf_paths(value, prefix + [str(key)]):
                yield item
    elif isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
        yield ".".join(prefix), node
    else:
        yield ".".join(prefix), node


def dotted_key_names(node, prefix=None):
    """Yield the path of every mapping key whose OWN NAME contains a '.'.

    Such a key is unaddressable by the dotted-path resolver: the schema would report it ABSENT
    forever while the value sits in the file. This guard caught exactly that defect on first run.
    """
    prefix = prefix or []
    if isinstance(node, dict):
        for key, value in node.items():
            name = str(key)
            if "." in name:
                yield "/".join(prefix + [name])
            for item in dotted_key_names(value, prefix + [name]):
                yield item


def is_unpinned(value):
    return isinstance(value, str) and value.lstrip().startswith(UNPINNED_MARKER)


def reason_of(value):
    text = value.lstrip()[len(UNPINNED_MARKER):].strip()
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser(description="Mechanical guard on the stage -1 pre-registration.")
    parser.add_argument("path", nargs="?", default=str(Path(__file__).with_name("prereg.yaml")))
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print("FATAL: no such file: %s" % path, file=sys.stderr)
        return 2
    try:
        tree = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("FATAL: could not parse %s: %s" % (path, exc), file=sys.stderr)
        return 2
    if not isinstance(tree, dict):
        print("FATAL: %s does not parse to a mapping" % path, file=sys.stderr)
        return 2

    absent, unpinned, exempted = [], [], []
    for key in REQUIRED_KEYS:
        found, value = resolve(tree, key)
        if not found:
            (exempted if key in STRUCTURAL_EXEMPTIONS else absent).append(key)
        elif is_unpinned(value):
            target = exempted if key in STRUCTURAL_EXEMPTIONS else unpinned
            target.append((key, reason_of(value)))

    required = set(REQUIRED_KEYS)
    all_leaves = list(leaf_paths(tree))
    extra = [p for p, _ in all_leaves if p not in required]
    extra_unpinned = [p for p, v in all_leaves if p not in required and is_unpinned(v)]
    malformed = list(dotted_key_names(tree))

    blocking = bool(absent or unpinned or malformed)

    if args.json:
        print(json.dumps({
            "file": str(path),
            "required_key_count": len(REQUIRED_KEYS),
            "absent": absent,
            "unpinned": [{"key": k, "reason": r} for k, r in unpinned],
            "malformed_dotted_keys": malformed,
            "undeclared_extra_keys": extra,
            "undeclared_extra_unpinned": extra_unpinned,
            "structurally_exempted": [k for k, _ in exempted] if exempted and isinstance(exempted[0], tuple) else exempted,
            "complete": not blocking,
        }, indent=2))
        return 1 if blocking else 0

    print("=" * 96)
    print("check_prereg.py -- stage -1 pre-registration guard")
    print("file:          %s" % path)
    print("required keys: %d" % len(REQUIRED_KEYS))
    print("=" * 96)

    if absent:
        print("")
        print("ABSENT -- required by the schema, not present in the file (%d):" % len(absent))
        for key in absent:
            print("  [ABSENT]   %s" % key)

    if unpinned:
        print("")
        print("UNPINNED -- present but not pinned (%d):" % len(unpinned))
        width = max(len(k) for k, _ in unpinned)
        for key, reason in unpinned:
            if len(reason) > 140:
                reason = reason[:137] + "..."
            print("  [UNPINNED] %-*s  <- %s" % (width, key, reason))

    if malformed:
        print("")
        print("MALFORMED -- %d key name(s) contain a '.', which makes them unaddressable by the"
              " dotted-path schema (BLOCKING):" % len(malformed))
        for key in malformed:
            print("  [MALFORMED] %s" % key)

    if extra:
        print("")
        print("NOTE -- %d leaf key(s) in the file but not in the schema (non-blocking; the schema"
              " has drifted behind the artefact):" % len(extra))
        for key in extra[:20]:
            print("  [EXTRA]    %s" % key)
        if len(extra) > 20:
            print("  ... and %d more" % (len(extra) - 20))

    if extra_unpinned:
        print("")
        print("WARNING -- undeclared key(s) carrying an UNPINNED value (invisible to the schema):")
        for key in extra_unpinned:
            print("  [EXTRA-UNPINNED] %s" % key)

    print("")
    print("-" * 96)
    total = len(absent) + len(unpinned)
    if not blocking:
        print("RESULT: COMPLETE -- %d/%d required fields pinned. 0 absent, 0 unpinned, 0 malformed."
              % (len(REQUIRED_KEYS), len(REQUIRED_KEYS)))
        print("-" * 96)
        return 0
    print("RESULT: INCOMPLETE -- %d absent, %d unpinned, %d malformed, %d/%d required fields pinned."
          % (len(absent), len(unpinned), len(malformed), len(REQUIRED_KEYS) - total, len(REQUIRED_KEYS)))
    print("The pre-registration is NOT complete. Do not claim any field above is pinned.")
    print("-" * 96)
    return 1


if __name__ == "__main__":
    sys.exit(main())
