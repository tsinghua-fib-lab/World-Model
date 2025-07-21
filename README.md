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

| Title | Pub. & Date | Code/Project URL |
|---|---|---|
| [DreamerV3: Mastering diverse domains with world models](https://arxiv.org/abs/2301.04104) | arXiv 2023 | [![Star](https://img.shields.io/github/stars/danijar/dreamerv3.svg?style=social&label=Star)](https://github.com/danijar/dreamerv3) |
| [DreamerV2: Learning skillful behaviors from high-dimensional observations](https://arxiv.org/abs/2003.01016) | arXiv 2020 | [![Star](https://img.shields.io/github/stars/danijar/dreamerv2.svg?style=social&label=Star)](https://github.com/danijar/dreamerv2) |
| [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) | ICLR 2016 | |
| [World model based trajectory prediction with transformer](https://ieeexplore.ieee.org/document/10185918) | IEEE Trans. ITS 2023 | |
| [Memory-augmented control with world models](https://arxiv.org/abs/2307.01639) | arXiv 2023 | |
| [Deep reinforcement learning in a handful of trials using model-based value expansion](https://arxiv.org/abs/1805.12117) | ICLR 2018 | |
| [Recurrent world models facilitate policy evolution](https://proceedings.neurips.cc/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf) | NeurIPS 2018 | [![Website](https://img.shields.io/badge/Website-9cf)](https://worldmodels.github.io/) |
| [Learning latent dynamics for planning from pixels](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf) | ICML 2019 | [![Website](https://img.shields.io/badge/Website-9cf)](https://danijar.com/project/planet) |
| [Dream to control: Learning behaviors by latent imagination](https://arxiv.org/abs/1912.01603) | ICLR 2020 | |
| [Mastering atari with discrete world models](https://arxiv.org/abs/2010.02193) | ICLR 2021 | |
| [Temporal difference learning for model-based reinforcement learning](https://arxiv.org/abs/1905.10906) | arXiv 2019 | |
| [Deep reinforcement learning using model-based value expansion](https://arxiv.org/abs/1805.12117) | ICLR 2018 | |
| [Model-based reinforcement learning from sparse rewards](https://arxiv.org/abs/2002.04949) | ICML 2020 | |
| [Model-based deep reinforcement learning with an uncertainty-aware environment model](https://arxiv.org/abs/1802.05760) | ICML 2018 | |
| [Efficient exploration in reinforcement learning with model-based uncertainty estimation](https://arxiv.org/abs/1802.08277) | ICML 2018 | |
| [Model-based reinforcement learning with probabilistic ensembles](https://arxiv.org/abs/1802.03157) | ICLR 2019 | |
| [When to trust your model: Model-based policy optimization](https://arxiv.org/abs/1906.08253) | NeurIPS 2019 | |
| [Model-based reinforcement learning for robotics with latent state models](https://ieeexplore.ieee.org/document/9079986) | RAL 2020 | |
| [Model-based reinforcement learning with probabilistic neural networks](https://arxiv.org/abs/1906.08253) | ICML 2019 | |
| [Neural network dynamics for model-based deep reinforcement learning](https://arxiv.org/abs/1905.08867) | ICLR 2019 | |
| [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.nature.com/articles/nature25760) | Nature 2018 | |
| [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature16961) | Nature 2016 | |
| [Reinforcement learning for motor control with neural networks](https://arxiv.org/abs/1905.10906) | ICML 2019 | |
| [Monte Carlo tree search for deep reinforcement learning](https://arxiv.org/abs/1905.08867) | ICLR 2019 | |
| [LLM-assisted robot planning with language-based world models](https://arxiv.org/abs/2307.03901) | arXiv 2023 | |
| [RoboDreamer: Learning Compositional World Models for Robot Imagination](https://arxiv.org/abs/2404.12377) | arXiv 2024 | [![Website](https://img.shields.io/badge/Website-9cf)](https://robodreamer.github.io/) |
| [Vision-language world models for long-horizon robot manipulation](https://arxiv.org/abs/2404.09038) | arXiv 2024 | |
| [Large language model-based multi-modal world model](https://arxiv.org/abs/2310.03816) | arXiv 2023 | |
| [Can large language models be trained as world models?](https://arxiv.org/abs/2309.11717) | arXiv 2023 | |
| [Mind agent: A comprehensive foundation agent for language-based task planning](https://arxiv.org/abs/2308.10091) | arXiv 2023 | [![Star](https://img.shields.io/github/stars/PKU-YuanGroup/MindAgent.svg?style=social&label=Star)](https://github.com/PKU-YuanGroup/MindAgent) |
| [LLM-based robotic task planning with world models](https://arxiv.org/abs/2403.00392) | arXiv 2024 | |
| [S3: Social-network Simulation System with Large Language Model-Empowered Agents](https://arxiv.org/abs/2307.14984) | arXiv 2023 | |
| [Generative world models for autonomous driving](https://arxiv.org/abs/2311.00287) | arXiv 2023 | |
| [Virtualhome: Simulating household activities via interaction with objects and people](https://openaccess.thecvf.com/content_ICCV_2019/papers/Pu_VirtualHome_Simulating_Household_Activities_via_Interaction_with_Objects_and_People_ICCV_2019_paper.pdf) | ICCV 2019 | |
| [World-scale generative models for object-centric scene understanding](https://arxiv.org/abs/2403.01353) | arXiv 2024 | |
| [LLM-driven task-oriented robot manipulation](https://arxiv.org/abs/2404.09038) | arXiv 2024 | |
| [ALFWorld: Aligning text and embodied environments for learning agents](https://proceedings.mlr.press/v119/li20c/li20c.pdf) | ICML 2020 | [![Star](https://img.shields.io/github/stars/alfworld/alfworld.github.io.svg?style=social&label=Star)](https://github.com/alfworld/alfworld.github.io) |
