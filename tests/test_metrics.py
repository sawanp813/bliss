import torch

from bliss.catalog import FullCatalog
from bliss.metrics import BlissMetrics


def test_metrics():
    slen = 50
    slack = 1.0
    bliss_metrics = BlissMetrics(slack)

    true_locs = torch.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.2, 0.2], [0.1, 0.1]]])
    est_locs = torch.tensor([[[0.49, 0.49], [0.1, 0.1]], [[0.19, 0.19], [0.01, 0.01]]])
    true_galaxy_bools = torch.tensor([[[1], [0]], [[1], [1]]])
    est_galaxy_bools = torch.tensor([[[0], [1]], [[1], [0]]])

    d_true = {
        "n_sources": torch.tensor([1, 2]),
        "plocs": true_locs * slen,
        "galaxy_bools": true_galaxy_bools,
    }
    true_params = FullCatalog(slen, slen, d_true)

    d_est = {
        "n_sources": torch.tensor([2, 2]),
        "plocs": est_locs * slen,
        "galaxy_bools": est_galaxy_bools,
    }
    est_params = FullCatalog(slen, slen, d_est)

    results_metrics = bliss_metrics(true_params, est_params)
    precision = results_metrics["precision"]
    recall = results_metrics["recall"]
    avg_distance = results_metrics["avg_distance"]

    class_acc = results_metrics["class_acc"]
    conf_matrix = results_metrics["conf_matrix"]

    assert precision == 2 / (2 + 2)
    assert recall == 2 / 3
    assert class_acc == 1 / 2
    assert conf_matrix.eq(torch.tensor([[1, 1], [0, 0]])).all()
    assert avg_distance.item() == 50 * (0.01 + (0.01 + 0.09) / 2) / 2
