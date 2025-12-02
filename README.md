# TextJosher: A Transfer-Based Black-Box Attack Method Against Text Classifiers

TextJosher is a transfer-based black-box adversarial attack framework for text classifiers. It overcomes the non-differentiability of text by using a surrogate model to estimate gradients and identify critical tokens, then generates fluent and semantically-preserving adversarial examples through a multi-constrained decoding mechanism. Our method achieves high attack success rates on standard classifiers and demonstrates strong transferability to large language models like LLaMA-2 and Mistral.


✨ Key Features

Black-Box Attack: Operates without access to the target model's internal parameters or gradients.

Gradient Estimation: Uses a local surrogate model to approximate gradients for discrete text data.

Multi-Constraint Generation: Decodes adversarial text with a joint loss enforcing semantic and grammatical integrity.

High Transferability: Successfully attacks powerful, publicly-available LLMs via transfer.

📚 Citation
If you use TextJosher in your research, please cite our paper:

@article{**WANG2026122888**,
title = {TextJosher: A transfer-based black-box attack method Against text classifiers},
journal = {Information Sciences},
volume = {731},
pages = {122888},
year = {2026},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2025.122888},
url = {https://www.sciencedirect.com/science/article/pii/S0020025525010242},
author = {Peishuai Wang and Sicong Zhang and Yang Xu and Xinlong He and Weida Xu},
keywords = {Natural language processing, Surrogate model, Adversarial attack, Textual adversarial examples, Semantic similarity},
abstract = {Although deep neural networks have achieved significant success in various domains, they remain susceptible to attacks from carefully crafted adversarial examples in the field of text. Existing text classification adversarial attack methods often suffer from low attack success rates and poor quality of generated samples, mainly due to two challenges: (1) accurately identifying salient tokens that significantly influence model decisions; (2) misleading classifiers with minimal text modifications while preserving semantic meaning and grammaticality. To address these issues, we propose TextJosher, a text adversarial example generation framework for text classification based on transfer-based black-box attacks. To overcome the non-differentiability of discrete text, TextJosher employs a local surrogate model to estimate gradients and computes embedding-level saliency to identify critical tokens. Furthermore, to improve the quality and stealthiness of adversarial samples, we design a decoding mechanism that integrates a masked language model head with a multi-constraint loss, incorporating both semantic similarity and grammatical fluency. Extensive experiments on text classification demonstrate that TextJosher outperforms baselines in terms of success rate, semantic similarity, and fluency. Its adversarial examples also transfer to LLaMA-2-7B-Chat and Mistral-7B-Instruct under decision-only, zero-shot evaluation, achieving high attack success rates on these large language models.}
}
