"""Pure mocked tests for H4 Kaggle freshness gates; makes no Kaggle calls."""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


LAUNCHER = Path(__file__).with_name("h4_kaggle_stage_a.py")
SPEC = importlib.util.spec_from_file_location("h4_kaggle_stage_a_under_test", LAUNCHER)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


@dataclass
class Result:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def kernel_csv(refs: list[str]) -> str:
    rows = ["ref,title,author,lastRunTime,totalVotes"]
    rows.extend(f"{ref},title,sirwildpino,,0" for ref in refs)
    return "\n\n".join(rows) + "\n"


def dataset_csv(refs: list[str]) -> str:
    rows = ["ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating"]
    rows.extend(f"{ref},title,1,2026-09-16,0,0,0.0" for ref in refs)
    return "\n\n".join(rows) + "\n"


class OwnerInventoryFreshnessTests(unittest.TestCase):
    owner = "sirwildpino"

    def test_kernel_absent_from_complete_inventory(self) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []

        def fake(account: str, *args: str) -> Result:
            calls.append((account, args))
            self.assertEqual(args, ("kernels", "list", "--mine", "--csv", "--page-size", "200"))
            return Result(stdout=kernel_csv(["sirwildpino/another-kernel"]))

        with patch.object(MOD, "_kaggle", fake):
            MOD.assert_remote_ref_absent("acct3", "kernels", "sirwildpino/h4-rank48-stagea-v2")
        self.assertEqual(len(calls), 1)

    def test_existing_kernel_is_refused(self) -> None:
        with patch.object(MOD, "_kaggle", return_value=Result(
            stdout=kernel_csv(["sirwildpino/h4-rank48-stagea-v2"])
        )):
            with self.assertRaisesRegex(SystemExit, "already exists"):
                MOD.assert_remote_ref_absent("acct3", "kernels", "sirwildpino/h4-rank48-stagea-v2")

    def test_status_403_is_never_used_as_absence_proof(self) -> None:
        calls: list[tuple[str, ...]] = []

        def fake(_: str, *args: str) -> Result:
            calls.append(args)
            if args[1] == "status":
                return Result(returncode=1, stderr="403 Client Error: Forbidden")
            return Result(stdout=kernel_csv([]))

        with patch.object(MOD, "_kaggle", fake):
            MOD.assert_remote_ref_absent("acct3", "kernels", "sirwildpino/h4-rank48-stagea-v2")
        self.assertEqual(calls, [("kernels", "list", "--mine", "--csv", "--page-size", "200")])

    def test_inventory_command_403_is_rejected(self) -> None:
        with patch.object(MOD, "_kaggle", return_value=Result(
            returncode=1, stderr="403 Client Error: Forbidden"
        )):
            with self.assertRaisesRegex(SystemExit, "command failed"):
                MOD.assert_remote_ref_absent("acct3", "kernels", "sirwildpino/h4-rank48-stagea-v2")

    def test_kernel_capacity_refuses_truncated_listing(self) -> None:
        refs = [f"sirwildpino/kernel-{i}" for i in range(200)]
        with patch.object(MOD, "_kaggle", return_value=Result(stdout=kernel_csv(refs))):
            with self.assertRaisesRegex(SystemExit, "capacity"):
                MOD.assert_remote_ref_absent("acct3", "kernels", "sirwildpino/h4-rank48-stagea-v2")

    def test_continuation_marker_is_pagination_ambiguity(self) -> None:
        text = kernel_csv([]) + "Next Page Token: uncertain\n"
        with patch.object(MOD, "_kaggle", return_value=Result(stdout=text)):
            with self.assertRaisesRegex(SystemExit, "continuation marker"):
                MOD.assert_remote_ref_absent("acct3", "kernels", "sirwildpino/h4-rank48-stagea-v2")

    def test_bad_header_wrong_owner_and_duplicate_are_refused(self) -> None:
        cases = (
            ("ref,title\nsirwildpino/a,title\n", "unexpected CSV header"),
            (kernel_csv(["other-owner/a"]), "wrong-owner"),
            (kernel_csv(["sirwildpino/a", "sirwildpino/a"]), "duplicate"),
        )
        for text, message in cases:
            with self.subTest(message=message), patch.object(MOD, "_kaggle", return_value=Result(stdout=text)):
                with self.assertRaisesRegex(SystemExit, message):
                    MOD.assert_remote_ref_absent("acct3", "kernels", "sirwildpino/h4-rank48-stagea-v2")

    def test_dataset_short_first_page_proves_complete_absence(self) -> None:
        with patch.object(MOD, "_kaggle", return_value=Result(
            stdout=dataset_csv(["sirwildpino/existing-dataset"])
        )) as mocked:
            MOD.assert_remote_ref_absent("acct3", "datasets", "sirwildpino/siliconllm-h4-rank48-v2")
        self.assertEqual(mocked.call_args.args[1:], ("datasets", "list", "--mine", "--csv", "--page", "1"))

    def test_existing_dataset_is_refused(self) -> None:
        with patch.object(MOD, "_kaggle", return_value=Result(
            stdout=dataset_csv(["sirwildpino/siliconllm-h4-rank48-v2"])
        )):
            with self.assertRaisesRegex(SystemExit, "already exists"):
                MOD.assert_remote_ref_absent("acct3", "datasets", "sirwildpino/siliconllm-h4-rank48-v2")

    def test_dataset_exact_end_marker_only_after_full_page(self) -> None:
        full = dataset_csv([f"sirwildpino/dataset-{i}" for i in range(20)])
        replies = [Result(stdout=full), Result(stdout="No datasets found\n")]
        with patch.object(MOD, "_kaggle", side_effect=replies):
            MOD.assert_remote_ref_absent("acct3", "datasets", "sirwildpino/siliconllm-h4-rank48-v2")
        with patch.object(MOD, "_kaggle", return_value=Result(stdout="No datasets found\n")):
            with self.assertRaisesRegex(SystemExit, "unexpected non-CSV"):
                MOD.assert_remote_ref_absent("acct3", "datasets", "sirwildpino/siliconllm-h4-rank48-v2")

    def test_dataset_repeated_page_and_safe_bound_are_refused(self) -> None:
        full = dataset_csv([f"sirwildpino/dataset-{i}" for i in range(20)])
        with patch.object(MOD, "_kaggle", return_value=Result(stdout=full)):
            with self.assertRaisesRegex(SystemExit, "repeated a page"):
                MOD.assert_remote_ref_absent("acct3", "datasets", "sirwildpino/siliconllm-h4-rank48-v2")
        page_one = dataset_csv([f"sirwildpino/a-{i}" for i in range(20)])
        page_two = dataset_csv([f"sirwildpino/b-{i}" for i in range(20)])
        with patch.object(MOD, "MAX_DATASET_PAGES", 2), patch.object(
            MOD, "_kaggle", side_effect=[Result(stdout=page_one), Result(stdout=page_two)]
        ):
            with self.assertRaisesRegex(SystemExit, "safe page bound"):
                MOD.assert_remote_ref_absent("acct3", "datasets", "sirwildpino/siliconllm-h4-rank48-v2")


@dataclass
class Quota:
    time_used: timedelta
    total_time_allowed: timedelta


@dataclass
class QuotaResponse:
    gpu_quota: Quota | None


class GpuQuotaTests(unittest.TestCase):
    def test_direct_api_uses_neutral_home_and_restores_environment(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_ops = SimpleNamespace(
                _NEUTRAL_HOME=root / "neutral",
                config_dir=lambda account: root / account,
            )
            before = {
                "KAGGLE_CONFIG_DIR": "global-config",
                "HOME": "global-home",
                "USERPROFILE": "global-profile",
                "KAGGLE_API_TOKEN": "global-token",
                "KAGGLE_USERNAME": "global-user",
                "KAGGLE_KEY": "global-key",
            }
            with patch.dict(sys.modules, {"kaggle_ops": fake_ops}), patch.dict(MOD.os.environ, before, clear=True):
                with MOD._isolated_kaggle_credentials("acct3"):
                    self.assertEqual(os.environ["KAGGLE_CONFIG_DIR"], str(root / "acct3"))
                    self.assertEqual(os.environ["HOME"], str(root / "neutral"))
                    self.assertEqual(os.environ["USERPROFILE"], str(root / "neutral"))
                    self.assertFalse({"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"} & set(os.environ))
                self.assertEqual(dict(os.environ), before)

    def test_integer_second_duration_uses_in_memory_compatibility_path(self) -> None:
        original_calls: list[object] = []

        def original(value: object) -> object:
            original_calls.append(value)
            return value

        decode = MOD._whole_second_duration_compatible(original)
        self.assertEqual(decode("0s"), timedelta(0))
        self.assertEqual(decode("108000s"), timedelta(hours=30))
        self.assertEqual(decode("1.5s"), "1.5s")
        self.assertEqual(original_calls, ["1.5s"])

    def test_direct_api_quota_accepts_acct3_zero_used_of_thirty_hours(self) -> None:
        response = QuotaResponse(Quota(timedelta(0), timedelta(hours=30)))
        with patch.object(MOD, "_quota_view", return_value=response) as quota_view:
            report = MOD.check_quota("acct3")
        quota_view.assert_called_once_with("acct3")
        self.assertEqual(report["gpu_quota_source"], "direct_api_in_memory_duration_shim")
        self.assertEqual(report["gpu_quota_used_hours"], 0.0)
        self.assertEqual(report["gpu_quota_total_hours"], 30.0)
        self.assertEqual(report["gpu_quota_remaining_hours"], 30.0)

    def test_quota_below_three_hours_fails_closed(self) -> None:
        response = QuotaResponse(Quota(timedelta(hours=27, minutes=1), timedelta(hours=30)))
        with patch.object(MOD, "_quota_view", return_value=response):
            with self.assertRaisesRegex(SystemExit, "3.0 h required"):
                MOD.check_quota("acct3")

    def test_missing_or_inconsistent_gpu_quota_is_refused(self) -> None:
        cases = (
            QuotaResponse(None),
            QuotaResponse(Quota(timedelta(hours=31), timedelta(hours=30))),
        )
        for response in cases:
            with self.subTest(response=response), patch.object(MOD, "_quota_view", return_value=response):
                with self.assertRaisesRegex(SystemExit, "quota response"):
                    MOD.check_quota("acct3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
