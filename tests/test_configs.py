def test_basic_configs(get_dataset, train_sleep):
    # test creating star model using config files.
    overrides = dict(model="basic_sleep_star_one_tile", training="cpu", dataset="cpu")
    sleep_net, _ = train_sleep(overrides)
    dataset = get_dataset(overrides)
    image_decoder = sleep_net.image_decoder
    image_encoder = sleep_net.image_encoder
    assert image_encoder.n_bands == image_decoder.n_bands == 1
    assert image_encoder.n_galaxy_params == image_decoder.n_galaxy_params == 8
    assert image_decoder.tile_slen == image_encoder.tile_slen
    assert image_decoder.prob_galaxy == 0.0
    assert dataset.n_batches == 1
