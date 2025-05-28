<div align="center">

# Minute-Long Videos with Dual Parallelisms

<!-- <img src='./assets/DualParal.png' width='80%' /> -->
<br>
<a href="https://arxiv.org/abs/2505.21070"><img src="https://img.shields.io/badge/ariXv-2505.21070-A42C25.svg" alt="arXiv"></a>
<a  href="https://dualparal-project.github.io/dualparal.github.io/"><img src="https://img.shields.io/badge/ProjectPage-DualParal-376ED2#376ED2.svg"></a>
</div>

## 📚 TL;DR (Too Long; Didn't Read)
**DualParal** is a distributed inference strategy for Diffusion Transformers (DiT)-based video diffusion models. It achieves high efficiency by parallelizing both temporal frames and model layers with the help of *block-wise denoising scheme*.
Feel free to visit our [paper](https://arxiv.org/abs/2505.21070) for more information.

## 🎥 Demo--more video samples in our [project page](https://dualparal-project.github.io/dualparal.github.io/)!
<div align="center">
    <img src="assets/gif1.gif" style="width: 416px; height: 240px; object-fit: cover;"/>
    <p style="text-align: justify; font-size: 10px; line-height: 1.2; margin: 5px 0;">
        A white-suited astronaut with a gold visor spins in dark space, tethered by a drifting cable. Stars twinkle around him as Earth glows blue in the distance. His suit reflects faint starlight against the vastness of the cosmos.
    </p>
    <img src="assets/gif2.gif" style="width: 416px; height: 240px; object-fit: cover;"/>
    <p style="text-align: justify; font-size: 10px; line-height: 1.2; margin: 5px 0;">
        A flock of birds glides through the warm sunset sky, wings outstretched. Their feathers catch golden light as they soar above silhouetted treetops, with the sky glowing in soft hues of amber and pink.
    </p>
</div>

## 🛠️ Setup
```
conda create -n DualParal python=3.10
conda activate DualParal
# Ensure torch >= 2.4.0 according to your cuda version, the following use CUDA12.1 as example
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 🚀 Usage
### **Quick Start —— DualParal on multiple GPUs with Wan2.1-1.3B (480p)**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m examples.DualParal_Wan --sample_steps 50 --num_per_block 8 --latents_num 40 --num_cat 8
```

### **Major parameters**
- **Basic Args**

| Parameter   | Description                            |
| ----------- | -------------------------------------- |
| `dtype`   | Model dtype (float64, float32, float16, fp32, fp16, half, bf16)       |
| `seed` | The seed to use for generating the video. |
| `save_file` | The file to save the generated video to. |
| `verbose` | Enable verbose mode for debug. |
| `export_image` | Enable exporting video frames. |

- **Model Args**

| Parameter   | Description                            |
| ----------- | -------------------------------------- |
| `model_id`   | Model Id for Wan-2.1 Video (Wan-AI/Wan2.1-T2V-1.3B-Diffusers, or Wan-AI/Wan2.1-T2V-14B-Diffusers).      |
| `height` | Height of generating videos. |
| `width` | Width of generating videos. |
| `sample_steps` | The sampling steps. |
| `sample_shift` | Sampling shift factor for flow matching schedulers. |
| `sample_guide_scale` | Classifier free guidance scale. |

- **Major Args for DualParal**

| Parameter   | Description                            |
| ----------- | -------------------------------------- |
| `prompt` | The prompt to generate the video from. |
| `num_per_block` | The number of latents per block in DualParal. |
| `latents_num` | The total number of latents sampled from video. `latents_num` **must** be divisible by `num_per_block`. The total number of video frames is calculated as (`latents_num` - 1) $\times$ 4 + 1. |
| `num_cat` | The number of latents to concatenate in previous and subsequent blocks separately. Increasing it (not greater than `num_per_block`) will lead better global consistency and temperoal coherence. Note that $Num_C$ in paper is equal to 2*`num_cat`.  |

### Further experiments
- **Original Wan implementation with single GPU**
```bash
python -m examples.Wan-Video.py 
```

- **DualParal on multiple GPUs with Wan2.1-14B (720p)**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m examples.DualParal_Wan --model_id Wan-AI/Wan2.1-T2V-14B-Diffusers --height 720 --width 1280 --sample_steps 50 --num_per_block 8 --latents_num 40 --num_cat 8
```

## ☀️ Acknowledgements
Our project is based on the [Wan2.1](https://github.com/Wan-Video/Wan2.1) model. We would like to thank the authors for their excellent work! ❤️

## 🔗 Citation
```
@misc{wang2025minutelongvideosdualparallelisms,
      title={Minute-Long Videos with Dual Parallelisms}, 
      author={Zeqing Wang and Bowen Zheng and Xingyi Yang and Yuecong Xu and Xinchao Wang},
      year={2025},
      eprint={2505.21070},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.21070}, 
}
```
