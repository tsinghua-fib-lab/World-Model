# Awesome-World-Model [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of awesome resources on World Models, based on the comprehensive survey "Understanding World or Predicting Future? A Comprehensive Survey of World Models".

![Loading Outline](asset/outline.png "outline")

## News🔥

* **[2024/11/21]** Initial release of our survey is available on [arXiv](https://arxiv.org/abs/2411.14499v1).
* **[2025/06/13]** Our survey paper "Understanding World or Predicting Future? A Comprehensive Survey of World Models" has been accepted by [ACM Computing Surveys](https://dl.acm.org/doi/abs/10.1145/3746449).
* **[2025/06/25]** Second version of our survey is available on [arXiv](https://arxiv.org/abs/2411.14499).
* **[2025/07/18]** Initial release of the Awesome-World-Model GitHub repository.

## Contact
If you have any suggestions or find our work helpful, feel free to contact us  
Email: dingjt15@tsinghua.org.cn

If this list helps your research, please ⭐ and cite:

```bibtex
@article{ding2025worldmodels,
  title={Understanding World or Predicting Future? A Comprehensive Survey of World Models},
  author={Ding, Jingtao and Zhang, Yunke and Shang, Yu and Zhang, Yuheng and Zong, Zefang and Feng, Jie and Yuan, Yuan and Su, Hongyu and Li, Nian and Sukiennik, Nicholas and Xu, Fengli and Li, Yong},
  journal={ACM Computing Surveys},
  year={2025}
}
```

## Table of Contents 🍃

* [1 Introduction & 2 Background](#1-introduction--2-background)
* [3 Implicit Representation of the External World](#3-implicit-representation-of-the-external-world)
    * [3.1 World Model in Decision Making](#31-world-model-decision-making)
    * [3.2 World Knowledge Learned by Models](#32-world-knowledge-learned)
* [4 Future Predictions of the External World](#4-future-predictions-of-the-external-world)
    * [4.1 What to Predict?](#41-what-to-predict)
    * [4.2 How to Predict?](#42-how-to-predict)
* [5 Applications of World Models](#5-applications-of-world-models)
    * [5.1 World Models for Decision-Making and Control](#51-world-models-for-decision-making-and-control)
    * [5.2 World Models for Generative Tasks](#52-world-models-for-generative-tasks)
    * [5.3 World Models for Embodied Intelligence](#53-world-models-for-embodied-intelligence)
* [6 Open Problems and Future Directions](#6-open-problems-and-future-directions)
* [7 Conclusion](#7-conclusion)

## 1 Introduction & 2 Background

| Title | Pub. & Date | Code/Project URL |
|---|---|---|
| [Sora: Creating video from text](https://openai.com/sora) | OpenAI 2024 | |
| [A path towards autonomous machine intelligence version 0.9.2, 2022-06-27](https://openreview.net/forum?id=BZ5a1r-kVsf) | Open Review 2022 | |
| [A framework for representing knowledge](https://dspace.mit.edu/handle/1721.1/5834) | 1974 | |
| [Recurrent world models facilitate policy evolution](https://proceedings.neurips.cc/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf) | 2018 | [![Website](https://img.shields.io/badge/Website-9cf)](https://worldmodels.github.io/) |
| [World models](https://arxiv.org/abs/1803.10122) | 2018 | [![Website](https://img.shields.io/badge/Website-9cf)](https://worldmodels.github.io/) |
| [Mental models: Towards a cognitive science of language, inference, and consciousness](https://www.hup.harvard.edu/books/9780674568512) | | |
| [Sora as an agi world model? a complete survey on text-to-video generation](https://arxiv.org/abs/2403.05131) | arXiv 2024 | |
| [Is sora a world simulator? a comprehensive survey on general world models and beyond](https://arxiv.org/abs/2405.03520) | arXiv 2024 | |
| [World models for autonomous driving: An initial survey](https://arxiv.org/abs/2401.01312) | IEEE T-IV 2024 | |
| [Data-centric evolution in autonomous driving: A comprehensive survey of big data system, data mining, and closed-loop technologies](https://arxiv.org/abs/2401.12888) | arXiv 2024 | |
| [Forging vision foundation models for autonomous driving: Challenges, methodologies, and opportunities](https://arxiv.org/abs/2401.08045) | arXiv 2024 | |
| [From Efficient Multimodal Models to World Models: A Survey](https://arxiv.org/abs/2407.00118) | arXiv 2024 | |
| [MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving](https://ieeexplore.ieee.org/document/8569113) | arXiv 2018 | |
| [YOLOP: You Only Look Once for Panoptic Driving Perception](https://link.springer.com/article/10.1007/s11633-022-1339-y) | Image and Vision Computing 2022 | [![Star](https://img.shields.io/github/stars/hustvl/YOLOP.svg?style=social&label=Star)](https://github.com/hustvl/YOLOP) |
| [Scene transformer: A unified multi-task model for behavior prediction and planning](https://openreview.net/forum?id=53A9Pu06k63) | | |
| [Motion transformer with global intention localization and local movement refinement](https://proceedings.neurips.cc/paper_files/paper/2022/file/2c89109d44ed535a8f05c15b25357c18-Paper-Conference.pdf) | NeurIPS 2022 | [![Star](https://img.shields.io/github/stars/sshaoshuai/MTR.svg?style=social&label=Star)](https://github.com/sshaoshuai/MTR) |
| [Query-centric trajectory prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf) | CVPR 2023 | |
| [Gnm: A general navigation model to drive any robot](https://ieeexplore.ieee.org/document/10160295) | IEEE Trans. on Robotics 2023 | [![Website](https://img.shields.io/badge/Website-9cf)](https://general-navigation-models.github.io/) |
| [Repvit: Revisiting mobile cnn from vit perspective](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.pdf) | CVPR 2024 | [![Star](https://img.shields.io/github/stars/THU-MIG/RepViT.svg?style=social&label=Star)](https://github.com/THU-MIG/RepViT) |
| [Learning latent dynamics for planning from pixels](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf) | ICML 2019 | [![Website](https://img.shields.io/badge/Website-9cf)](https://danijar.com/project/planet) |
| [S3: Social-network Simulation System with Large Language Model-Empowered Agents](https://arxiv.org/abs/2307.14984) | arXiv 2023 | |
| [Generative agents: Interactive simulacra of human behavior](https://dl.acm.org/doi/10.1145/3586183.3606763) | UIST 2023 | [![Star](https://img.shields.io/github/stars/joonspk-research/generative_agents.svg?style=social&label=Star)](https://github.com/joonspk-research/generative_agents) |

## 3 Implicit Representation of the External World

### 3.1 World Model in Decision Making

| Title | Pub. & Date | Code/Project URL |
|---|---|---|
| [Deep reinforcement learning in a handful of trials using probabilistic dynamics models](https://proceedings.neurips.cc/paper/2018/file/006f52e9102a8d3be2fe5614f42ba989-Paper.pdf) | NeurIPS 2018 | [![Star](https://img.shields.io/github/stars/kchua/handful-of-trials.svg?style=social&label=Star)](https://github.com/kchua/handful-of-trials) |
| [PWM: Policy Learning with Multi-Task World Models](https://openreview.net/pdf?id=KKN4OFyzvN) | OpenReview | [![Website](https://img.shields.io/badge/Website-9cf)](https://imgeorgiev.com/pwm) |
| [Recurrent world models facilitate policy evolution](https://proceedings.neurips.cc/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf) | NeurIPS 2018 | [![Website](https://img.shields.io/badge/Website-9cf)](https://worldmodels.github.io/) |
| [Dream to control: Learning behaviors by latent imagination](https://arxiv.org/abs/1912.01603) | arXiv 2019 | [![Website](https://img.shields.io/badge/Website-9cf)](https://danijar.com/project/dreamer) |
| [Leveraging pre-trained large language models to construct and utilize world models for model-based task planning](https://proceedings.neurips.cc/paper_files/paper/2023/file/a7f14143a825f3e0c036c641829e0839-Paper-Conference.pdf) | NeurIPS 2023 | [![Star](https://img.shields.io/github/stars/GuanSuns/LLMs-World-Models-for-Planning.svg?style=social&label=Star)](https://github.com/GuanSuns/LLMs-World-Models-for-Planning) |
| [Mastering atari with discrete world models](https://arxiv.org/abs/2010.02193) | arXiv 2020 | [![Website](https://img.shields.io/badge/Website-9cf)](https://danijar.com/project/dreamerv2) |
| [Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-024-07871-x) | Nature 2024 | [![Website](https://img.shields.io/badge/Website-9cf)](https://danijar.com/project/dreamerv3) |
| [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU) | OpenReview | [![Website](https://img.shields.io/badge/Website-9cf)](https://nicklashansen.github.io/td-mpc2) |
| [When to trust your model: Model-based policy optimization](https://proceedings.neurips.cc/paper/2019/file/258be2de6eb6952aa35065474166379a-Paper.pdf) | NeurIPS 2019 | |
| [Offline reinforcement learning as one big sequence modeling problem](https://proceedings.neurips.cc/paper/2021/file/099fe6b0b444c23836c4a5d07346082b-Paper.pdf) | NeurIPS 2021 | [![Star](https://img.shields.io/github/stars/jannerm/trajectory-transformer.svg?style=social&label=Star)](https://github.com/jannerm/trajectory-transformer) |
| [Model predictive control](https://link.springer.com/book/10.1007/978-3-319-22592-7) | Springer | |
| [Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees](https://arxiv.org/abs/1807.03858) | arXiv 2018 | |
| [Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning](https://ieeexplore.ieee.org/document/8460505) | IEEE 2018 | |
| [A game theoretic framework for model based reinforcement learning](https://proceedings.mlr.press/v119/rajeswaran20a.html) | PMLR 2020 | [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/mbrl-game) |
| [General agents need world models](https://arxiv.org/abs/2506.01622) | arXiv 2025 | |
| [Mastering memory tasks with world models](https://arxiv.org/abs/2403.04253) | arXiv 2024 | |
| [A generalist dynamics model for control](https://arxiv.org/abs/2305.10912) | arXiv 2023 | |
| [Exploring model-based planning with policy networks](https://arxiv.org/abs/1906.08649) | arXiv 2019 | |
| [Derivative-free optimization via classification](https://ojs.aaai.org/index.php/AAAI/article/view/10255) | AAAI | |
| [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961) | Nature 2016 | |
| [Mastering the game of go without human knowledge](https://www.nature.com/articles/nature24270) | Nature 2017 | |
| [A0c: Alpha zero in continuous action space](https://arxiv.org/abs/1805.09613) | arXiv 2018 | |
| [Value prediction network](https://proceedings.neurips.cc/paper/2017/file/d045c59a90d7587d8d671b5f5aec4e7c-Paper.pdf) | NeurIPS 2017 | |
| [Probabilistic adaptation of text-to-video models](https://arxiv.org/abs/2306.01872) | arXiv 2023 | [![Website](https://img.shields.io/badge/Website-9cf)](https://prob-t2v.github.io/) |
| [RoboDreamer: Learning Compositional World Models for Robot Imagination](https://arxiv.org/abs/2404.12377) | arXiv 2024 | [![Website](https://img.shields.io/badge/Website-9cf)](https://robodreamer.github.io/) |
| [Discuss before moving: Visual language navigation via multi-expert discussions](https://ieeexplore.ieee.org/document/10565013) | IEEE 2024 | [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/discussnav) |
| [OVER-NAV: Elevating Iterative Vision-and-Language Navigation with Open-Vocabulary Detection and Structured Representation](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_OVER-NAV_Elevating_Iterative_Vision-and-Language_Navigation_With_Open-Vocabulary_Detection_and_CVPR_2024_paper.pdf) | CVPR 2024 | [![Star](https://img.shields.io/github/stars/GAN-Z/OVER-NAV.svg?style=social&label=Star)](https://github.com/GAN-Z/OVER-NAV) |
| [RILA: Reflective and Imaginative Language Agent for Zero-Shot Semantic Audio-Visual Navigation](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_RILA_Reflective_and_Imaginative_Language_Agent_for_Zero-Shot_Semantic_CVPR_2024_paper.pdf) | CVPR 2024 | [![Website](https://img.shields.io/badge/Website-9cf)](https://rila-agent.github.io/) |
| [Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models](https://arxiv.org/abs/2501.09686) | arXiv 2025 | |
| [Position: LLMs can't plan, but can help planning in LLM-modulo frameworks](https://arxiv.org/abs/2402.01817) | arXiv 2024 | |
| [Language models meet world models: Embodied experiences enhance language models](https://proceedings.neurips.cc/paper_files/paper/2023/file/206f20295c520334237c15815653b216-Paper-Conference.pdf) | NeurIPS 2023 | [![Star](https://img.shields.io/github/stars/E2WM/E2WM-NeurIPS23.svg?style=social&label=Star)](https://github.com/E2WM/E2WM-NeurIPS23) |
| [Virtualhome: Simulating household activities via programs](https://openaccess.thecvf.com/content_cvpr_2018/papers/Puig_VirtualHome_Simulating_Household_CVPR_2018_paper.pdf) | CVPR 2018 | [![Website](https://img.shields.io/badge/Website-9cf)](http://virtual-home.org/) |
| [Learning to Model the World with Language](https://arxiv.org/abs/2308.01399) | arXiv 2023 | [![Website](https://img.shields.io/badge/Website-9cf)](https://dynalang.github.io/) |
| [Reason for Future, Act for Now: A Principled Framework for Autonomous LLM Agents with Provable Sample Efficiency](https://arxiv.org/abs/2309.17382) | arXiv 2023 | |
| [Alfworld: Aligning text and embodied environments for interactive learning](https://openreview.net/forum?id=0IO2UCN-kDb) | OpenReview | [![Star](https://img.shields.io/github/stars/alfworld/alfworld.svg?style=social&label=Star)](https://github.com/alfworld/alfworld) |
