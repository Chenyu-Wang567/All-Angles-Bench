
# 🔥 📊 QuickStart 📊 🔥
# Use VLMEvalKit to Evaluate Open-Sourced MLLMs on All-Angles Bench 
We adopt and extend [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/main) to support our All-Angles Bench. Below we list step-by-step instructions to clone benchmark and setup the recommended conda environments for evaluating different MLLMs. Some MLLMs may fail under mismatched CUDA/PyTorch or transformers versions, so we recommend isolating each group in its own environment.


## **🤗 Step 1: Clone All-Angles Bench from HuggingFace Hub**
To begin, clone the benchmark locally under VLMEvalkit.
```bash
$ conda install git-lfs
$ git lfs install

$ cd VLMEvalkit
$ git lfs clone https://huggingface.co/datasets/ch-chenyu/All-Angles-Bench
```
🔥🔥 **[Important]** 🔥🔥 

To complete the full preprocessing pipeline required for benchmark evaluation, visit the [**All-Angles Bench on HuggingFace🤗**](https://huggingface.co/datasets/ch-chenyu/All-Angles-Bench) and follow the steps.


## **🛠️ Step2: Install Recommended Conda Environments for Open-Sourced MLLMs:**
Note that some MLLMs may not be able to run under certain environments, we recommend the following conda environment settings under separate groups to evaluate corresponding MLLMs:

- **Please use** `envs/vlm_eval_qwen2s.yml` **for**: `Qwen2.5-VL series`, `InternVL2.5 series`, `Ovis2 series`, `LLaVA-OneVision` series, and `LLaVA-Video-Qwen2` series.
- **Please use** `envs/vlm_eval_deepseek.yml` **for**: `Deepseek-VL2 series`.
- **Please use** `envs/vlm_eval_cambrian.yml` **for**: `Cambrian series`.

Below are instruction examples for creating the conda environment that you prefer
```bash
# For Qwen2.5-VL, InternVL2.5, Ovis2, LLaVA-OneVision, LLaVA-Video-Qwen2..
conda env create -f envs/vlm_eval_qwen2s.yml && conda activate vlm_eval_qwen2s

# For Deepseek-VL2..
conda env create -f envs/vlm_eval_deepseek.yml && conda activate vlm_eval_deepseek

# For Cambrian..
conda env create -f envs/vlm_eval_cambrian.yml && conda activate vlm_eval_cambrian
```


## **📝 Step3: Evaluate MLLMs on All-Angles Bench:**

Once your environment is ready, launch the evaluation using:
```bash
# For vlm_eval_qwen2s env, run the following example
python run.py --data All_Angles_Bench --model Qwen2.5-VL-3B-Instruct --verbose

# You might run the following command if across 4 GPUs on the same node
torchrun --nproc-per-node=4 run.py --data All_Angles_Bench --model Qwen2.5-VL-3B-Instruct --verbose
```
The evaluation results would be shown on both terminal and under `outputs/Qwen2.5-VL-3B-Instruct/` as csv file.


**Note:** You can also switch the `--model` argument to other supported MLLMs such as `Qwen2.5-VL-7B-Instruct`, `InternVL2_5-4B`, or `Ovis2-4B` when using the `vlm_eval_qwen2s` environment. Please check `vlmeval/config.py` to select more MLLMs. 

We currently support the following MLLM series:  
`Qwen2.5-VL` series, `InternVL2.5` series, `Ovis2` series, `Deepseek-VL2` series, `Cambrian` series, `LLaVA-OneVision` series, and `LLaVA-Video-Qwen2` series.

For detailed configuration instructions, please refer to [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/main).


## 🌟 Citation

If you find this repo helpful, please consider giving **star🌟**, and citing the following works.

```bib
@article{yeh2025seeing,
  title={Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs},
  author={Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Ruoyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao and Yi Ma},
  journal={arXiv preprint arXiv:2504.15280},
  year={2025}
}
```


```bib
@inproceedings{duan2024vlmevalkit,
  title={Vlmevalkit: An open-source toolkit for evaluating large multi-modality models},
  author={Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={11198--11201},
  year={2024}
}
```
