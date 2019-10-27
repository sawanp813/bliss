import torch

from torch.distributions import normal, categorical

def get_is_on_from_n_stars(n_stars, max_stars):
    batchsize = len(n_stars)
    is_on_array = torch.zeros((batchsize, max_stars), dtype = torch.long).to(device)
    for i in range(max_stars):
        is_on_array[:, i] = (n_stars > i)

    return is_on_array

def sample_class_weights(class_weights, n_samples = 1):
    """
    draw a sample from Categorical variable with
    probabilities class_weights
    """

    # draw a sample from Categorical variable with
    # probabilities class_weights

    cat_rv = categorical.Categorical(probs = class_weights)
    return cat_rv.sample((n_samples, )).detach().squeeze()

def sample_normal(mean, logvar):
    return mean + torch.exp(0.5 * logvar) * torch.randn(mean.shape).to(device)

def get_one_hot_encoding_from_int(z, n_classes):
    z = z.long()

    assert len(torch.unique(z)) <= n_classes

    z_one_hot = torch.zeros(len(z), n_classes).to(device)
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot

def get_categorical_loss(log_probs, one_hot_encoding):
    assert torch.all(log_probs <= 0)
    assert log_probs.shape[0] == one_hot_encoding.shape[0]
    assert log_probs.shape[1] == one_hot_encoding.shape[1]

    return torch.sum(
        -log_probs * one_hot_encoding, dim = 1)

def _logit(x, tol = 1e-8):
    return torch.log(x + tol) - torch.log(1 - x + tol)

def eval_normal_logprob(x, mu, log_var):
    return - 0.5 * log_var - 0.5 * (x - mu)**2 / (torch.exp(log_var) + 1e-5) - 0.5 * np.log(2 * np.pi)

def eval_logitnormal_logprob(x, mu, log_var):
    logit_x = _logit(x)
    return eval_normal_logprob(logit_x, mu, log_var)

def eval_lognormal_logprob(x, mu, log_var, tol = 1e-8):
    log_x = torch.log(x + tol)
    return eval_normal_logprob(log_x, mu, log_var)
