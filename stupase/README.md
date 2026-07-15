# StuPASE: Towards Low-Hallucination and Studio-Quality Generative Speech Enhancement
[![arxiv](https://img.shields.io/badge/arXiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2603.09234)
[![demo](https://img.shields.io/badge/Demo-orange?logo=audacity&logoColor=white)](https://xiaobin-rong.github.io/stupase_demo/)
[![models](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/cisco-ai/stupase)

🎉 This is the official implementation of our Interspeech 2026 paper: 
[StuPASE: Towards Low-Hallucination Studio-Quality Generative Speech Enhancement](https://arxiv.org/abs/2603.09234)

## Pretrained Checkpoints 
Three checkpoints are provided:
- `DeWavLM-R.pt`: the semantic enhancement module
- `CFM.pt`: the flow-matching-based acoustic enhancement module
- `Vocoder_Mel-16k.pt`: the vocoder for reconstructing waveforms from Mel-spectrogram representations

The list of training data used for the released checkpoint is as follows:
- **Speech**: LibriVox (DNS5), LibriTTS, VCTK, LibriSpeech
- **Noise**: DNS5
- **RIRs**: OpenSLR26, OpenSLR28

The speech and RIR portions are the same as in the original paper, but the noise part excludes three corpora: WHAM!, FSD50K, and FMA.


The performance of the retrained version compared to the original one:
| Model | DNSMOS | UTMOS | SBS | LPS | SpkSim | WER (%) |
|:-----:|:------:|:-----:|:---:|:---:|:------:|:-------:|
| DeWavLM-R (orig.) | 3.35 | 3.94 | 0.84 | 0.88 | 0.49 | 13.22 |
| **DeWavLM-R (retrained)** | 3.38 | 3.62 | 0.85 | 0.89 | 0.41 | 12.27 |
| StuPASE (orig.) | 3.37  | 4.08  | 0.85 |0.90 |0.68    | 11.57|
| **StuPASE (retrained)** | 3.36 | 4.02 | 0.85 | 0.89 | 0.66 | 12.06 |

It can be seen that the retrained version achieves performance very close to that of the original version on our simulated test set.

## Inference
To run inference on audio files, ensure you are in the `stupase` directory and use:


```bash
python -m inference.inference -I <input_dir> -O <output_dir> [options]
```

| Argument       | Requirement / Default | Description                                                                 |
|----------------|-----------------------|-----------------------------------------------------------------------------|
| `-I` (`--input_dir`)  | **required**          | Path to the input directory containing audio files.                   |
| `-O` (`--output_dir`) | **required**          | Path to the output directory where enhanced files will be saved.      |
| `-D` (`--device`)     | default: `cuda:0`     | Torch device to run inference on, e.g., `cuda:0`, `cuda:1`, or `cpu`. |
| `-E` (`--extension`)  | default: `.wav`       | Audio file extension to process.                                      |

Audio examples can be found in `../test/audio_enh_stupase`.

## Training
### Step 1: Train a Mel-based vocoder
- training script: `train/train_melvocoder.py`
- training configuration: `configs/cfg_train_melvocoder.yaml`

    ```bash
    python -m train.train_melvocoder -C configs/cfg_train_melvocoder.yaml -D 0,1
    ```
- inference script: `inference/infer_melvocoder.py`
    ```bash
    python -m inference.infer_melvocoder -C configs/cfg_infer.yaml -D 0
    ```

**Purpose**: Pre-train a Mel-based vocoder, which will be used for final waveform reconstruction.

### Step 2: Finetuning DeWavLM
- training script: `train/train_dewavlm.py`
- training configuration: `configs/cfg_train_dewavlm.yaml`
- inference script: `inference/infer_dewavlm.py`

(Usage is the same as in Step 1.)

**Purpose**: Fine-tune the semantic enhancement module (DeWavLM) using dry targets to improve dereverberation performance.

### Step 3: Training a flow-matching module
- training script: `train/train_flow.py`
- training configuration: `configs/cfg_train_flow.yaml`
- inference script: `inference/infer_flow.py`

(Usage is the same as in Step 1.)

**Purpose**: Train the DiT-based flow-matching module for acoustic enhancement. This module takes noisy Mel-spectrograms and enhanced WavLM-L24 representations as inputs and produces enhanced Mel-spectrograms.

## Creating Checkpoints
Once all training steps are completed, the corresponding checkpoints can be prepared for inference:
- `utils/create_ckpt_wavlm.py` is used to create a DeWavLM-R checkpoint.
- `utils/create_ckpt.py` is used to create a Vocoder_Mel_16k or CFM checkpoint.

## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{StuPASE,
      title={{StuPASE: Towards Low-Hallucination Studio-Quality Generative Speech Enhancement}}, 
      author={Xiaobin Rong and Jun Gao and Zheng Wang and Mansur Yesilbursa and Kamil Wojcicki and Jing Lu},
      year={2026},
      eprint={2603.09234},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2603.09234}, 
}
```

## Contact
Xiaobin Rong: [xiaobin.rong@smail.nju.edu.cn](mailto:xiaobin.rong@smail.nju.edu.cn)

Mansur Yesilbursa: [myesilbu@cisco.com](myesilbu@cisco.com)