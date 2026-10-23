import torch

from rsl_rl.datasets.supermimic_rsi import ReturnAwareReferenceSampler


class _Dataset:
    reference_bins = [[(0, index)] for index in range(4)]

    def sample_reference_states_by_bin(self, bin_ids):
        return {"selected": bin_ids.clone()}


def test_sampler_returns_reference_states_and_bin_ids_with_return_aware_mixture() -> None:
    sampler = ReturnAwareReferenceSampler(_Dataset(), num_bins=4)
    sampler.update_returns(torch.arange(4), torch.tensor([0.0, 1.0, 2.0, 3.0]))

    states, bin_ids = sampler.sample_reference_states(torch.arange(20_000))

    assert torch.equal(states["selected"], bin_ids)
    assert sampler.probabilities.shape == (4,)
    assert torch.allclose(sampler.probabilities.sum(), torch.tensor(1.0))
    assert torch.isfinite(sampler.probabilities).all()
    assert (bin_ids == 0).float().mean() > 0.5


def test_equal_returns_and_unobserved_bins_produce_finite_probabilities() -> None:
    sampler = ReturnAwareReferenceSampler(_Dataset(), num_bins=4)
    sampler.update_returns(torch.tensor([0, 1]), torch.tensor([2.0, 2.0]))

    assert torch.isfinite(sampler.probabilities).all()
    assert torch.allclose(sampler.probabilities.sum(), torch.tensor(1.0))
    assert torch.all(sampler.probabilities > 0)


def test_sampler_checkpoint_restores_statistics_and_distribution() -> None:
    sampler = ReturnAwareReferenceSampler(_Dataset(), num_bins=4)
    sampler.update_returns(torch.tensor([0, 1, 2, 3]), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    state = sampler.state_dict()
    restored = ReturnAwareReferenceSampler(_Dataset(), num_bins=4)
    restored.load_state_dict(state)

    assert torch.equal(restored.probabilities, sampler.probabilities)
    assert torch.equal(restored.return_counts, sampler.return_counts)
