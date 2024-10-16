# VQVAE-2
Generating Diverse High-Fidelity Images with VQ-VAE-2 [Work in Progress]
PyTorch implementation of Hierarchical, Vector Quantized, Variational Autoencoders (VQ-VAE-2) from the paper "Generating Diverse High-Fidelity Images with VQ-VAE-2"

Original paper can be found here

Vector Quantizing layer based off implementation by @rosinality found here.

Aiming for a focus on supporting an arbitrary number of VQ-VAE "levels". Most implementations in PyTorch typically only use 2 which is limiting at higher resolutions. This repository contains checkpoints for a 3-level and 5-level VQ-VAE-2, trained on FFHQ1024.

This project will not only contain the VQ-VAE-2 architecture, but also an example autoregressive prior and latent dataset extraction.

This project is very much Work-in-Progress. VQ-VAE-2 model is mostly complete. PixelSnail prior models are still experimental and most definitely do not work.

Usage
VQ-VAE-2 Usage

Run VQ-VAE-2 training using the config task_name found in hps.py. Defaults to cifar10:

python main-vqvae.py --task task_name

Evaluate VQ-VAE-2 from parameters state_dict_path on task task_name. Defaults to cifar10:

python main-vqvae.py --task task_name --load-path state_dict_path --evaluate
Other useful flags:

--no-save       # disables saving of files during training

--cpu           # do not use GPU

--batch-size    # overrides batch size in cfg.py, useful for evaluating on larger batch size

--no-tqdm       # disable tqdm status bars

--no-save       # disables saving of files

--no-amp        # disables using native AMP (Automatic Mixed Precision) operations

--save-jpg      # save all images as jpg instead of png, useful for extreme resolutions

Latent Dataset Generation

Run latent dataset generation using VQ-VAE-2 saved at path that was trained on task task_name. Defaults to cifar10:

python main-latents.py path --task task_name

Result is saved in latent-data directory.

Other useful flags:

--cpu           # do not use GPU
--batch-size    # overrides batch size in cfg.py, useful for evaluating on larger batch size
--no-tqdm       # disable tqdm status bars
--no-save       # disables saving of files
--no-amp        # disables using native AMP (Automatic Mixed Precision) operations
Discrete Prior Usage
Run level level PixelSnail discrete prior training using the config task_name found in hps.py using latent dataset saved at path latent_dataset.pt and VQ-VAE vqvae_path to dequantize conditioning variables. Defaults to cifar10:

python main-pixelsnail.py latent_dataset.pt vqvae_path.pt level(0) --task task_name
Other useful flags:

--cpu           # do not use GPU
--load-path     # resume from saved state on disk
--batch-size    # overrides batch size in cfg.py, useful for evaluating on larger batch size
--save-jpg      # save all images as jpg instead of png, useful for extreme resolutions
--no-tqdm       # disable tqdm status bars
--no-save       # disables saving of files
Sample Generation
Run sampling script on trained VQ-VAE-2 and PixelSnail priors using the config task_name (default cifar10) found in hps.py. The first positional argument is the path to the VQ-VAE-2 checkpoint. The remaining L positional arguments are the PixelSnail prior checkpoints from level 0 to L.

python main-sample.py vq_vae_path.pt pixelsnail_0_path.pt pixel_snail_1_path.pt ... --task task_name
Other useful flags:

--cpu           # do not use GPU
--batch-size    # overrides batch size in cfg.py, useful for evaluating on larger batch size
--nb-samples    # number of samples to generate. defaults to 1.
--no-tqdm       # disable tqdm status bars
--no-save       # disables saving of files
--no-amp        # disables using native AMP (Automatic Mixed Precision) operations
--save-jpg      # save all images as jpg instead of png, useful for extreme resolutions
--temperature   # controls softmax temperature during sampling
