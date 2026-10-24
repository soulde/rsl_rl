import torch

from rsl_rl.datasets.supermimic_rsi import ReturnAwareReferenceSampler


class _Dataset:
    reference_bins = [[(0, index)] for index in range(4)]

    def sample_reference_states_by_bin(self, bin_ids):
        return {"selected": bin_ids.clone()}


def test_sampler_returns_reference_states_and_bin_ids_with_return_aware_mixture() -> None:
    dataset = _Dataset()
    sampler = ReturnAwareReferenceSampler(num_bins=4)
    sampler.update_returns(torch.arange(4), torch.tensor([0.0, 1.0, 2.0, 3.0]))

    batch = sampler.sample(dataset, 20_000)
    states, bin_ids = batch.states, batch.bin_ids

    assert torch.equal(states["selected"], bin_ids)
    assert sampler.probabilities.shape == (4,)
    assert torch.allclose(sampler.probabilities.sum(), torch.tensor(1.0))
    assert torch.isfinite(sampler.probabilities).all()
    assert (bin_ids == 0).float().mean() > 0.5


def test_equal_returns_and_unobserved_bins_produce_finite_probabilities() -> None:
    sampler = ReturnAwareReferenceSampler(num_bins=4)
    sampler.update_returns(torch.tensor([0, 1]), torch.tensor([2.0, 2.0]))

    assert torch.isfinite(sampler.probabilities).all()
    assert torch.allclose(sampler.probabilities.sum(), torch.tensor(1.0))
    assert torch.all(sampler.probabilities > 0)


def test_sampler_checkpoint_restores_statistics_and_distribution() -> None:
    sampler = ReturnAwareReferenceSampler(num_bins=4)
    sampler.update_returns(torch.tensor([0, 1, 2, 3]), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    state = sampler.state_dict()
    restored = ReturnAwareReferenceSampler(num_bins=4)
    restored.load_state_dict(state)

    assert torch.equal(restored.probabilities, sampler.probabilities)
    assert torch.equal(restored.return_estimates, sampler.return_estimates)


def test_sampler_matches_reference_return_and_historical_ema() -> None:
    sampler = ReturnAwareReferenceSampler(num_bins=4, max_return=100.0)
    sampler.update_returns(torch.tensor([0, 1]), torch.tensor([10.0, 20.0]))

    expected_returns = torch.tensor([2.0, 4.0, 0.0, 0.0])
    torch.testing.assert_close(sampler.return_estimates, expected_returns)
    soft = torch.softmax(-25.0 * expected_returns / 100.0, dim=0)
    initial_history = torch.full((4,), 1.0 / 4)
    sampler.sample(_Dataset(), 8)

    torch.testing.assert_close(sampler.historical_weights, 0.999 * initial_history + 0.001 * soft)
    expected = (
        0.5 * torch.full((4,), 0.25) + 0.4 * soft + 0.1 * sampler.historical_weights / sampler.historical_weights.sum()
    )
    torch.testing.assert_close(sampler.probabilities, expected)


def test_ordinary_reference_bin_excludes_last_clip_frame():
    class Dataset:
        device = "cpu"
        reference_bins = [[(0, 0), (0, 1)]]
        motions = [{"joint_pos": torch.zeros(2, 1)}]

        def _decode_reference_states(self, frames):
            return {}

    dataset = Dataset()
    sampler = ReturnAwareReferenceSampler(1)
    states = sampler.sample(dataset, 64).states
    assert not states["reference_frame_ids"].any()
