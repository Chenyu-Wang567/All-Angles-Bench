<h1>
  <img src="static/all-angles-bench-icon.png" alt="Icon" style="height: 24px; vertical-align: middle;"/>
  Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs
</h1>


[Chun-Hsiao Yeh*](https://danielchyeh.github.io/), [Chenyu Wang*](https://scholar.google.com/citations?user=ZkCLeicAAAAJ&hl=en), [Shengbang Tong](https://tsb0601.github.io/), [Ta-Ying Cheng](https://ttchengab.github.io/), [Ruoyu Wang](https://scholar.google.com/citations?user=V5H0P28AAAAJ), [Tianzhe Chu](https://tianzhechu.com/), [Yuexiang Zhai](https://yx-s-z.github.io/), [Yubei Chen](https://yubeichen.com/), [Shenghua Gao](https://svip-lab.github.io/), [Yi Ma](https://people.eecs.berkeley.edu/~yima/).(*Equal Contribution)

UC Berkeley, TranscEngram, NYU, University of Oxford, UC Davis, HKU



-----
<a href='https://next-gpt.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/pdf/2401.10727'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>  
<a href='https://arxiv.org/pdf/2401.10727'><img src='https://img.shields.io/badge/Arxiv-Page-purple'></a> 
<a href="https://huggingface.co/collections/tianzhechu/sftvsrl-models-and-data-6797ba6de522c7de7fcb80ba"><img src='https://img.shields.io/badge/Data-HuggingFace-red'></a>

<p align="center" width="100%">
<a target="_blank"><img src="static/figure-teaser.png" alt="teaser" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

📌 **A Benchmark for Multi-View Understanding:** We introduce <i>All-Angles Bench</i>, a large-scale benchmark with over 2,100 human-annotated multi-view QA pairs across 90 real-world scenes.

📊 **Performance Evaluation:** We benchmark 27 leading MLLMs, including Gemini-2.0-Flash, Claude-3.7-Sonnet, and GPT-4o. Our results reveal a substantial gap between MLLMs and human.

🔍 **Decoding MLLM Shortcomings:** We identify two major failure modes in MLLMs: (1) weak cross-view correspondence under occlusions and (2) poor estimation of coarse camera poses.


## 🎉 News 
- [x] [2025.04] 📢📢 Release the All-Angles Benchmark on HuggingFace.
- [x] [2025.04] 📢📢 Upload paper and init project.


## To Do List
- [ ] Evaluation Code Release
- [x] All-Angles Benchmark Release
- [x] Project Page & Technical Report


## All-Angles Bench
<p align="center" width="100%">
<a target="_blank"><img src="static/figure-all-angles-bench.png" alt="all-angles-bench" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

**Benchmark Overview:** We introduce All-Angles Bench, a benchmark designed to evaluate the multi-view reasoning capabilities of MLLMs, containing <b>2,132</b> question-answer pairs carefully annotated across <b>90</b> diverse real-world scenes sourced from EGO4D-EXO and EgoHumans. All-Angles Bench comprises six challenging tasks including <b><i>counting, attribute identification, relative distance, relative direction, manipulation, and camera pose estimation</i></b>.These question types are designed to investigate several major aspects of 3D scene understanding, from creating correspondence between objects to associating relative object and camera poses.


## Contact

For any questions or feedback, feel free to contact [Chun-Hsiao Yeh](daniel_yeh@berkeley.edu) and [Chenyu Wang](chenyuwang5562@gmail.com).

## Citation

 If you find All-Angles Bench useful in your research or applications, please kindly cite:
```
@article{yeh2025seeing,
  title={Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs},
  author={Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Rouyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao and Yi Ma},
  journal={arXiv preprint arXiv:2401.10727},
  year={2025}
}
```




## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, 
[EgoHumans](https://github.com/rawalkhirodkar/egohumans),
[Ego-Exo4D](https://github.com/facebookresearch/Ego4d),
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
Thanks for their wonderful work and data.



## License Notices
This repository is under MIT License.
All-Angles Bench is a research project intended for non-commercial use only. 
One must NOT use the code of All-Angles Bench for any illegal, harmful, violent, racist, or sexual purposes. 
One is strictly prohibited from engaging in any activity that will potentially violate these guidelines.
Any potential commercial use of this code should be approved by the authors.
