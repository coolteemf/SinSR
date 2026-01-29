#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Volume Super-Resolution using SinSR

This script loads a medical volume (e.g., NIfTI), extracts 2D slices,
applies SinSR super-resolution to each slice, and reassembles them
into a super-resolved volume with adjusted affine matrix.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf

from sampler import Sampler
from basicsr.utils.download_util import load_file_from_url


def get_configs(colab=False):
    """Load SinSR configuration and prepare checkpoints."""
    if colab:
        configs = OmegaConf.load('/content/SinSR/configs/SinSR.yaml')
    else:
        configs = OmegaConf.load('./configs/SinSR.yaml')
    
    # Prepare checkpoint directory
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    
    # Download SinSR checkpoint if needed
    ckpt_path = ckpt_dir / 'SinSR_v1.pth'
    if not ckpt_path.exists():
        load_file_from_url(
            url="https://github.com/wyf0912/SinSR/releases/download/v1.0/SinSR_v1.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
        )
    
    # Download VQGAN checkpoint if needed
    vqgan_path = ckpt_dir / 'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
        load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
        )
    
    # Configure model paths and parameters
    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = 15
    configs.diffusion.params.sf = 4
    configs.autoencoder.ckpt_path = str(vqgan_path)
    
    return configs


def normalize_slice(slice_data: np.ndarray, volume_data_min: float, volume_data_max: float) -> np.ndarray:
    """
    Normalize slice data to [0, 1] range.
    
    Args:
        slice_data: 2D numpy array with arbitrary range
        
    Returns:
        Normalized 2D array in [0, 1] range
    """
    data_min = slice_data.min()
    data_max = slice_data.max()
    
    if data_max - data_min < 1e-10:
        # Handle constant slices
        normalized = np.zeros_like(slice_data)
    else:
        normalized = (slice_data - volume_data_min) / (volume_data_max - volume_data_min)
    
    return normalized


def slice_to_rgb(slice_data: np.ndarray) -> np.ndarray:
    """
    Convert 2D slice to RGB image format.
    
    Args:
        slice_data: 2D numpy array in [0, 1] range
        
    Returns:
        3D numpy array (H, W, 3) in [0, 1] range
    """
    # Stack the 2D slice three times to create RGB
    rgb_slice = np.stack([slice_data, slice_data, slice_data], axis=-1)
    return rgb_slice


def rgb_to_slice(rgb_slice: np.ndarray) -> np.ndarray:
    """
    Convert RGB slice back to 2D grayscale.
    
    Args:
        rgb_slice: 3D numpy array (H, W, 3) in [0, 1] range
        
    Returns:
        2D numpy array in [0, 1] range (averaged RGB channels)
    """
    # Average RGB channels to get grayscale
    return rgb_slice.mean(axis=-1)


def load_volume(volume_path: str) -> tuple:
    """
    Load a medical volume and return data with metadata.
    
    Args:
        volume_path: Path to NIfTI volume file
        
    Returns:
        Tuple of (volume_data, affine, header)
    """
    volume_obj = nib.load(volume_path)
    volume_data = volume_obj.get_fdata()
    affine = volume_obj.affine
    header = volume_obj.header
    
    return volume_data, affine, header


def save_volume(volume: np.ndarray, filename: str, affine: Optional[np.ndarray] = None, 
                header: Optional[nib.Nifti1Header] = None) -> None:
    """
    Save a volume to NIfTI file.
    
    Args:
        volume: 3D numpy array
        filename: Output filename
        affine: Affine transformation matrix (default: identity)
        header: NIfTI header object (optional)
    """
    if affine is None:
        affine = np.eye(4)
    
    # Ensure correct file extension
    if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):
        filename += '.nii.gz'
    
    img = nib.Nifti1Image(volume, affine, header=header)
    nib.save(img, filename)
    print(f"Saved: {filename}")


def superresolve_slice_tensor(sampler: Sampler, slice_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply SinSR super-resolution to a slice tensor.
    
    Args:
        sampler: SinSR Sampler instance
        slice_tensor: 1 x 3 x H x W tensor in [0, 1] range (RGB)
        
    Returns:
        Super-resolved tensor in [0, 1] range
    """
    # Convert from [0, 1] to [-1, 1] as expected by the model
    slice_tensor_norm = (slice_tensor - 0.5) / 0.5
    
    # Apply super-resolution
    sr_tensor = sampler.sample_func(
        slice_tensor_norm,
        noise_repeat=False,
        one_step=True,  # SinSR uses single step
        apply_decoder=True
    )
    
    # Convert back from [-1, 1] to [0, 1]
    sr_tensor = sr_tensor * 0.5 + 0.5
    
    return sr_tensor.clamp(0.0, 1.0)


def process_volume_sinsr(
    volume_path: str,
    output_path: str,
    sampling_axis: int = -1,
    sf: int = 4,
    colab: bool = False,
    seed: int = 12345,
    chop_size: int = 256,
    chop_stride: int = 224
) -> None:
    """
    Process a volume using SinSR super-resolution.
    
    Args:
        volume_path: Path to input NIfTI volume
        output_path: Path for output NIfTI volume
        sampling_axis: Axis along which to extract slices (default: last axis)
        sf: Super-resolution scaling factor (default: 4)
        colab: Whether running in Google Colab (affects paths)
        seed: Random seed for reproducibility
        chop_size: Size of patches for processing large images
        chop_stride: Stride for patch overlap
    """
    print(f"Loading volume from: {volume_path}")
    
    # Load the volume
    volume_data, affine, header = load_volume(volume_path)
    volume_data_min = volume_data.min()
    volume_data_max = volume_data.max()
    print(f"Volume shape: {volume_data.shape}")
    print(f"Value range: [{volume_data_min:.4f}, {volume_data_max:.4f}]")
    
    # Initialize SinSR model
    print("\nInitializing SinSR model...")
    configs = get_configs(colab=colab)
    sampler = Sampler(
        configs,
        chop_size=chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=seed,
    )
    
    # Get the number of slices along the specified axis
    num_slices = volume_data.shape[sampling_axis]
    print(f"\nProcessing {num_slices} slices...")
    
    # Storage for super-resolved slices
    sr_slices = []
    
    # Extract and process each slice
    for i in range(num_slices):
        # Extract slice along the specified axis
        if sampling_axis == -1 or sampling_axis == 2:
            slice_data = volume_data[..., i]
        elif sampling_axis == 0:
            slice_data = volume_data[i, :, :]
        elif sampling_axis == 1:
            slice_data = volume_data[:, i, :]
        else:
            raise ValueError(f"Invalid sampling_axis: {sampling_axis}")
        
        # Normalize to [0, 1]
        normalized_slice = normalize_slice(slice_data, volume_data_min, volume_data_max)
        
        # Convert to RGB
        rgb_slice = slice_to_rgb(normalized_slice)
        
        # Convert to tensor and add batch dimension
        slice_tensor = torch.from_numpy(rgb_slice).permute(2, 0, 1).unsqueeze(0).float().cuda()
        
        # Apply super-resolution
        sr_tensor = superresolve_slice_tensor(sampler, slice_tensor)
        
        # Convert back to numpy and extract RGB
        sr_rgb = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Convert back to grayscale
        sr_slice = rgb_to_slice(sr_rgb)
        
        sr_slices.append(sr_slice)
        
        if (i + 1) % 10 == 0 or i == num_slices - 1:
            print(f"Processed {i + 1}/{num_slices} slices")
    
    # Stack super-resolved slices
    print("\nAssembling super-resolved volume...")
    sr_slices = np.stack(sr_slices, axis=-1)
    
    # Re-scale back to original value range
    original_min = volume_data.min()
    original_max = volume_data.max()
    sr_volume = sr_slices * (original_max - original_min) + original_min
    
    print(f"Super-resolved volume shape: {sr_volume.shape}")
    print(f"Super-resolved value range: [{sr_volume.min():.4f}, {sr_volume.max():.4f}]")
    
    # Adjust affine matrix for super-resolution
    print("\nAdjusting affine matrix...")
    sr_affine = affine.copy()
    # The upsampling affects the first two dimensions (height and width)
    # We divide the corresponding column(s) by the scaling factor
    sr_affine[0, 0] /= sf  # Width scaling
    sr_affine[1, 1] /= sf  # Height scaling
    
    # Save the super-resolved volume
    print(f"\nSaving super-resolved volume to: {output_path}")
    save_volume(sr_volume, output_path, affine=sr_affine, header=header)
    
    print("\nVolume super-resolution complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Apply SinSR super-resolution to medical volumes (NIfTI)'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input NIfTI volume'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path for output NIfTI volume'
    )
    parser.add_argument(
        '--axis',
        type=int,
        default=-1,
        choices=[0, 1, 2, -1],
        help='Axis along which to extract slices (0: sagittal, 1: coronal, 2/-1: axial, default: -1)'
    )
    parser.add_argument(
        '--sf',
        type=int,
        default=4,
        help='Super-resolution scaling factor (default: 4)'
    )
    parser.add_argument(
        '--colab',
        action='store_true',
        help='Running in Google Colab (adjusts paths)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=12345,
        help='Random seed for reproducibility (default: 12345)'
    )
    parser.add_argument(
        '--chop_size',
        type=int,
        default=256,
        help='Patch size for processing large images (default: 256)'
    )
    parser.add_argument(
        '--chop_stride',
        type=int,
        default=224,
        help='Patch stride for overlap (default: 224)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Process the volume
    process_volume_sinsr(
        volume_path=args.input,
        output_path=args.output,
        sampling_axis=args.axis,
        sf=args.sf,
        colab=args.colab,
        seed=args.seed,
        chop_size=args.chop_size,
        chop_stride=args.chop_stride
    )


if __name__ == "__main__":
    main()