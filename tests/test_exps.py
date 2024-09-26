from exps.run_narrow_passage_clearance_trial import narrow_passage_clearance

import numpy as np


class TestRadiusBound:
    test_rng = np.random.default_rng(seed=1999)
    d = 2
    delta = 0.499
    sample_schedule = [100, 1000]
    max_radius = 1.0
    n_trials = 100

    def test_increasing_samples_does_not_destroy_solution(self):
        bad_seeds = []
        for _ in range(20):
            trial_seed = self.test_rng.integers(0, 2 ** 32)
            trial_df = narrow_passage_clearance(
                self.delta,
                self.d,
                rng_seed=trial_seed,
                sample_schedule=self.sample_schedule,
                prm_type="radius",
                max_connections=self.max_radius
            )
            # pick off whether a sol was found for 100 or 1000
            small_set_has_sol = not np.isnan(
                trial_df[trial_df['n_samples'] == self.sample_schedule[0]]['conn_ub'].to_list()[0]
            )
            large_set_has_sol = not np.isnan(
                trial_df[trial_df['n_samples'] == self.sample_schedule[-1]]['conn_ub'].to_list()[0]
            )

            if small_set_has_sol and not large_set_has_sol:
                bad_seeds.append(trial_seed)

        assert not bad_seeds, "Found seeds where more samples broke solvability."

    # def test_bad_seed(self):
    #     bad_seed = 2018866881
    #     trial_df = narrow_passage_clearance(
    #         self.delta,
    #         self.d,
    #         rng_seed=bad_seed,
    #         sample_schedule=self.sample_schedule,
    #         prm_type="radius",
    #         max_connections=self.max_radius
    #     )
    #     # pick off whether a sol was found for 100 or 1000
    #     small_set_has_sol = not np.isnan(
    #         trial_df[trial_df['n_samples'] == self.sample_schedule[0]]['conn_ub'].to_list()[0]
    #     )
    #     large_set_has_sol = not np.isnan(
    #         trial_df[trial_df['n_samples'] == self.sample_schedule[-1]]['conn_ub'].to_list()[0]
    #     )
    #     assert not small_set_has_sol or large_set_has_sol, "failed (small_set_has_sol => bigger_set_has_sol)"
