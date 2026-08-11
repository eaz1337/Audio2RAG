import numpy as np
import pytest
from fakes import FakeEmbedder

from transform.embed import BgeM3Embedder, Embedder


class _NotAnEmbedder:
    pass


def test_fake_embedder_satisfies_the_protocol():
    assert isinstance(FakeEmbedder(), Embedder)


def test_unrelated_object_does_not_satisfy_the_protocol():
    assert not isinstance(_NotAnEmbedder(), Embedder)


def test_fake_embedder_output_shape_matches_dim():
    embedder = FakeEmbedder(dim=16)

    vectors = embedder.embed(["hello", "world"])

    assert vectors.shape == (2, 16)
    assert vectors.dtype == np.float32


def test_fake_embedder_is_deterministic_across_calls():
    embedder = FakeEmbedder(dim=8)

    first = embedder.embed(["some chunk text"])
    second = embedder.embed(["some chunk text"])

    np.testing.assert_array_equal(first, second)


def test_fake_embedder_is_deterministic_across_instances():
    first = FakeEmbedder(dim=8).embed(["some chunk text"])
    second = FakeEmbedder(dim=8).embed(["some chunk text"])

    np.testing.assert_array_equal(first, second)


def test_fake_embedder_differs_by_text():
    embedder = FakeEmbedder(dim=8)

    vectors = embedder.embed(["chunk one", "chunk two"])

    assert not np.array_equal(vectors[0], vectors[1])


def test_fake_embedder_preserves_batch_order():
    embedder = FakeEmbedder(dim=8)

    batch = embedder.embed(["alpha", "beta"])
    solo_alpha = embedder.embed(["alpha"])
    solo_beta = embedder.embed(["beta"])

    np.testing.assert_array_equal(batch[0], solo_alpha[0])
    np.testing.assert_array_equal(batch[1], solo_beta[0])


@pytest.mark.slow
def test_bge_m3_embedder_produces_normalised_dense_vectors():
    embedder = BgeM3Embedder()

    vectors = embedder.embed(["Zdanie testowe po polsku.", "A test sentence in English."])

    assert embedder.name == "BAAI/bge-m3"
    assert embedder.dim == vectors.shape[1]
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)
