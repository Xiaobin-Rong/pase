# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from models.pase import PASE

REPO_ID = "cisco-ai/pase"

def get_checkpoint_path(ckpt_arg, filename, download_dir=None):
    """
    Ensures the availability of a checkpoint file by resolving local paths 
    or downloading from the HuggingFace Hub.

    Args:
        ckpt_arg (str): Local path override. If it exists, use this path.
        filename (str): Filename to retrieve from the remote repository.
        download_dir (str, optional): Custom download destination. 

    Returns:
        str: Absolute path to the checkpoint.
    """
    
    if ckpt_arg and os.path.exists(ckpt_arg):
        full_path = os.path.abspath(ckpt_arg)
        print(f"[*] Using user-specified local checkpoint: {full_path}")
        return full_path
    
    print(f"[*] Downloading {filename} from Hugging Face ({REPO_ID})...")
    
    path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=filename, 
        local_dir=download_dir,
        local_dir_use_symlinks=False
    )

    absolute_path = os.path.abspath(path)
    print(f"[*] Checkpoint is stored at: {absolute_path}")
    
    return absolute_path


def inference_file(input_file, output_file, model):
    """
    Run inference on a single audio file and save the result.
    Args:
        input_file (str): Path to input audio file.
        output_file (str): Path to save enhanced audio.
        model: Initialized model for inference.
    """
    audio, fs = sf.read(input_file, dtype='float32')
    input_tensor = torch.FloatTensor(audio).unsqueeze(
        0).to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model(input_tensor)
    enhanced = output.cpu().detach().numpy().squeeze()
    
    scale = np.max(np.abs(audio))
    enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8) * scale
    
    sf.write(output_file, enhanced, fs)


def inference_folder(input_dir, output_dir, model, extension='.wav'):
    """
    Run inference on all audio files in a folder and save results to output_dir.
    Args:
        input_dir (str): Directory with input audio files.
        output_dir (str): Directory to save enhanced files.
        model: Initialized model for inference.
        extension (str): File extension to filter (default: '.wav').
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in tqdm(os.listdir(input_dir)):
        if fname.lower().endswith(extension):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            inference_file(in_path, out_path, model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run PASE inference on audio files.")
    parser.add_argument('-I', '--input_dir', type=str, required=True,
                        help='Input directory with audio files')
    parser.add_argument('-O', '--output_dir', type=str, required=True,
                        help='Output directory for enhanced files')
    parser.add_argument('-D', '--device', type=str, default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('-E', '--extension', type=str, default='.wav',
                        help='Audio file extension (default: .wav)')
    
    parser.add_argument('--dewavlm_ckpt', type=str, default=None, 
                        help='Path to DeWavLM.pt (if None, download from HF)')
    parser.add_argument('--vocoder_ckpt', type=str, default=None, 
                        help='Path to Vocoder_Dual.pt (if None, download from HF)')
    parser.add_argument('--download_dir', type=str, default=None,
                        help='Directory to download checkpoints (if None, use HF default cache directory)')
    
    args = parser.parse_args()
    
    resolved_dewavlm_path = get_checkpoint_path(
        args.dewavlm_ckpt, "DeWavLM.pt", args.download_dir
    )
    resolved_vocoder_path = get_checkpoint_path(
        args.vocoder_ckpt, "Vocoder_Dual.pt", args.download_dir
    )

    device = torch.device(args.device)
    model = PASE(
        dewavlm_ckpt_path=resolved_dewavlm_path,
        dewavlm_output_layer=[1, 24],
        vocoder_ckpt_path=resolved_vocoder_path,
    ).to(device).eval()

    inference_folder(args.input_dir, args.output_dir,
                     model, extension=args.extension)