import torch
from hydra.utils import instantiate


def prepare_image(x, device):
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device=device)
    # image dimensions must be a multiple of 16
    height = x.size(2) - (x.size(2) % 16)
    width = x.size(3) - (x.size(3) % 16)
    return x[:, :, :height, :width]


def predict(cfg):
    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    # load trained weights for encoder
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    encoder.eval()  # evaluation

    sdss = instantiate(cfg.predict.dataset)
    batch = {  # prepare batch of SDSS images for prediction
        # size: 1 x 1 x 1488 x 2048
        "images": prepare_image(sdss[0]["image"], cfg.predict.device),
        "background": prepare_image(sdss[0]["background"], cfg.predict.device),
        # size: same as images
    }

    with torch.no_grad():
        pred = encoder.encode_batch(batch)  # combines images/background
        est_cat = encoder.variational_mode(pred)

    print("{} light sources detected".format(est_cat.n_sources.item()))
