import math
import unittest

import numpy as np

from delta_tuning_framework.synthetic_adapter import (
    ExperimentConfig,
    DEFAULT_COLUMNS,
    activation_stats,
    run_experiment,
    run_one_configuration,
)


class SyntheticAdapterTests(unittest.TestCase):
    def test_identity_is_recoverable_at_adapter_rank(self):
        rows = run_one_configuration(
            seed=0,
            d=64,
            adapter_rank=8,
            intrinsic_dim=8,
            noise_sigma=0.0,
            activation="identity",
            n_train=512,
            n_test=128,
            replacement_ranks=[1, 8],
            cluster_counts=[2],
            output_classes=5,
        )
        matching = [
            row
            for row in rows
            if row["replacement_method"] == "lora_bias" and row["replacement_rank"] == 8
        ]
        self.assertEqual(len(matching), 1)
        self.assertLess(float(matching[0]["epsilon_test"]), 1e-10)

    def test_csv_schema_columns_are_stable(self):
        self.assertEqual(
            DEFAULT_COLUMNS,
            [
                "seed",
                "d",
                "adapter_rank",
                "intrinsic_dim",
                "noise_sigma",
                "activation",
                "replacement_method",
                "replacement_rank",
                "num_clusters",
                "epsilon_train",
                "epsilon_test",
                "r2_train",
                "r2_test",
                "teacher_student_fidelity_train",
                "teacher_student_fidelity_test",
                "activation_entropy_mean",
                "activation_entropy_std",
                "activation_active_fraction_mean",
                "activation_active_fraction_std",
            ],
        )

    def test_activation_stats_nan_for_non_switching_activations(self):
        stats = activation_stats(np.ones((10, 3)), "identity")
        self.assertTrue(math.isnan(stats["activation_entropy_mean"]))
        self.assertTrue(math.isnan(stats["activation_active_fraction_mean"]))

    def test_smoke_grid_row_count(self):
        config = ExperimentConfig(
            seeds=[0],
            n_train=128,
            n_test=64,
            hidden_dim=64,
            adapter_rank=8,
            intrinsic_dims=[4],
            noise_sigmas=[0.0],
            activations=["identity", "relu"],
            replacement_ranks=[1, 8],
            cluster_counts=[2],
        )
        rows = run_experiment(config, progress=False)
        # Per configuration: no_delta, bias_only, full_affine, 2 ranks for
        # low_rank_no_bias, 2 ranks for lora_bias, and 2 piecewise rows.
        self.assertEqual(len(rows), 2 * 9)


if __name__ == "__main__":
    unittest.main()
