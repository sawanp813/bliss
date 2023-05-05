"""Functions to evaluate the performance of BLISS predictions."""
from typing import Dict, Optional

import torch
from einops import rearrange, reduce
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torchmetrics import Metric
from tqdm import tqdm

from bliss.catalog import FullCatalog


class BlissMetrics(Metric):
    """Calculates aggregate detection metrics on batches over full images (not tiles)."""

    tp: Tensor
    fp: Tensor
    avg_distance: Tensor
    tp_gal: Tensor
    total_n_matches: Tensor
    total_coadd_n_matches: Tensor
    total_coadd_gal_matches: Tensor
    total_correct_class: Tensor
    conf_matrix: Tensor
    full_state_update: Optional[bool] = True

    def __init__(
        self,
        slack: float = 1.0,
        dist_sync_on_step: bool = False,
        disable_bar: bool = True,
    ) -> None:
        """Computes matches between true and estimated locations.

        Args:
            slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
            dist_sync_on_step: See torchmetrics documentation.
            disable_bar: Whether to show progress bar

        Attributes:
            tp: true positives = # of sources matched with a true source.
            fp: false positives = # of predicted sources not matched with true source
            avg_distance: Average l-infinity distance over matched objects.
            total_true_n_sources: Total number of true sources over batches seen.
            total_correct_class: Total # of correct classifications over matched objects.
            total_n_matches: Total # of matches over batches.
            conf_matrix: Confusion matrix (galaxy vs star)
            disable_bar: Whether to show progress bar
            total_n_matches: Total number of true matches.
            total_correct_class: Total number of correct classifications.
            Confusion matrix: Confusion matrix of galaxy vs. star
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.slack = slack
        self.disable_bar = disable_bar

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tp_gal", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("avg_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_true_n_sources", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_correct_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("conf_matrix", default=torch.tensor([[0, 0], [0, 0]]), dist_reduce_fx="sum")
        self.add_state("total_n_matches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_coadd_gal_matches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_correct_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("conf_matrix", default=torch.tensor([[0, 0], [0, 0]]), dist_reduce_fx="sum")

    # pylint: disable=no-member
    def update(self, true: FullCatalog, est: FullCatalog) -> None:  # type: ignore
        """Update the internal state of metrics including tp, fp, total_coadd_gal_matches, etc."""
        assert true.batch_size == est.batch_size

        count = 0
        desc = "Bliss Metric per batch"
        for b in tqdm(range(true.batch_size), desc=desc, disable=self.disable_bar):
            # number of true light sources, number of estimated light sources
            ntrue, nest = true.n_sources[b].int().item(), est.n_sources[b].int().item()
            # true light sources, estimated light sources
            tlocs, elocs = true.plocs[b], est.plocs[b]
            tgbool, egbool = true["galaxy_bools"][b].reshape(-1), est["galaxy_bools"][b].reshape(-1)
            if ntrue > 0 and nest > 0:
                # tlocs ~ true light sources, elocs ~ estimated light sources
                mtrue, mest, dkeep, avg_distance = match_by_locs(tlocs, elocs, self.slack)
                # number of GOOD matches, only keeps estimated sources that are in GOOD matches
                tp = len(elocs[mest][dkeep])
                # GOOD matched true light sources
                true_galaxy_bools = true["galaxy_bools"][b][mtrue][dkeep]
                # total number of GOOD matched true light sources
                tp_gal = true_galaxy_bools.bool().sum()
                # false positives - Number of estimated light sources included in BAD matches
                fp = nest - tp
                tgbool = tgbool[mtrue][dkeep].reshape(-1)
                egbool = egbool[mest][dkeep].reshape(-1)
                assert fp >= 0
                self.tp += tp
                self.tp_gal += tp_gal
                self.fp += fp
                self.avg_distance += avg_distance
                self.total_true_n_sources += ntrue  # type: ignore
                self.total_n_matches += len(egbool)
                self.total_coadd_gal_matches += tgbool.sum().int().item()
                self.total_correct_class += tgbool.eq(egbool).sum().int()
                self.conf_matrix += confusion_matrix(tgbool, egbool, labels=[1, 0])
                count += 1
        self.avg_distance /= count

    def compute(self) -> Dict[str, Tensor]:
        """Calculate f1, misclassification accuracy, confusion matrix."""
        precision = self.tp / (self.tp + self.fp)  # = PPV = positive predictive value
        recall = self.tp / self.total_true_n_sources  # = TPR = true positive rate
        f1 = (2 * precision * recall) / (precision + recall)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_distance": self.avg_distance,
            "n_galaxies_detected": self.tp_gal,
            "n_matches": self.total_n_matches,
            "n_matches_gal_coadd": self.total_coadd_gal_matches,
            "class_acc": self.total_correct_class / self.total_n_matches,
            "conf_matrix": self.conf_matrix,
        }


def match_by_locs(true_locs, est_locs, slack=1.0):
    """Match true and estimated locations and returned indices to match.

    Permutes `est_locs` to find minimal error between `true_locs` and `est_locs`.
    The matching is done with `scipy.optimize.linear_sum_assignment`, which implements
    the Hungarian algorithm.

    Automatically discards matches where at least one location has coordinates **exactly** (0, 0).

    Args:
        slack: Threshold for matching objects a `slack` l-infinity distance away (in pixels).
        true_locs: Tensor of shape `(n1 x 2)`, where `n1` is the true number of sources.
            The centroids should be in units of PIXELS.
        est_locs: Tensor of shape `(n2 x 2)`, where `n2` is the predicted
            number of sources. The centroids should be in units of PIXELS.

    Returns:
        A tuple of the following objects:
        - row_indx: Indicies of true objects matched to estimated objects.
        - col_indx: Indicies of estimated objects matched to true objects.
        - dist_keep: Matched objects to keep based on l1 distances.
        - avg_distance: Average l-infinity distance over matched objects.
    """
    assert len(true_locs.shape) == len(est_locs.shape) == 2
    assert true_locs.shape[-1] == est_locs.shape[-1] == 2
    assert isinstance(true_locs, torch.Tensor) and isinstance(est_locs, torch.Tensor)

    # reshape
    locs1 = true_locs.view(-1, 2)  # 11x2 - coordinates for true light sources
    locs2 = est_locs.view(-1, 2)  # 7x2 - coordinates for estimated light sources

    # remove (0,0) entries in estimated, true light sources
    # -----------
    locs1 = locs1[torch.abs(locs1).sum(dim=1) != 0]
    locs2 = locs2[torch.abs(locs2).sum(dim=1) != 0]
    # -----------

    # takes absolute pairwise difference along common dimension (i.e. 2), size: 11x7x2
    locs_abs_diff = (rearrange(locs1, "i j -> i 1 j") - rearrange(locs2, "i j -> 1 i j")).abs()
    # ixj matrix (e.g. 11x7), manhattan distance b/w each pair of true and estimated light sources
    locs_err = reduce(locs_abs_diff, "i j k -> i j", "sum")
    # taking maximum along k (l_infty norm)
    locs_err_l_infty = reduce(locs_abs_diff, "i j k -> i j", "max")

    # Penalize all pairs which are greater than slack apart to favor valid matches.
    # adds l_infty norm from locs_error to all light source pairs whose L1 distance is
    # greater than slack
    locs_err = locs_err + (locs_err_l_infty > slack) * locs_err.max()

    # find minimal permutation and return matches
    # convert light source error matrix to CSR - (COO, CSC not ideal))
    csr_locs_err = csr_matrix(locs_err.detach().cpu())
    row_indx, col_indx = min_weight_full_bipartite_matching(csr_locs_err)

    # we match objects based on distance too.
    # only match objects that satisfy threshold on l-infinity distance.
    # do not match fake objects with locs = (0, 0)

    # maximum linear distance between pairs of matched light sources (l-infty)
    dist = (locs1[row_indx] - locs2[col_indx]).abs().max(1)[0]

    # COND1 (GOOD match): makes sure that L-infinity distance is less than slack
    cond1 = (dist < slack).bool()

    dist_keep = cond1

    # ignore: average l-infinity distance over VALID matched objects. (all matches are now valid)
    avg_distance = dist.mean()

    if dist_keep.sum() > 0:
        assert dist[dist_keep].max() <= slack

    # return matching, avg l-infinity distance b/w matches and a boolean array indicating
    # the GOOD matches
    return row_indx, col_indx, cond1, avg_distance
