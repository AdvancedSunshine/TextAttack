# TextJosher: A Transfer-Based Black-Box Attack Method Against Text Classifiers

TextJosher is a transfer-based black-box adversarial attack framework for text classifiers. It overcomes the non-differentiability of text by using a surrogate model to estimate gradients and identify critical tokens, then generates fluent and semantically-preserving adversarial examples through a multi-constrained decoding mechanism. Our method achieves high attack success rates on standard classifiers and demonstrates strong transferability to large language models like LLaMA-2 and Mistral.


✨ Key Features

Black-Box Attack: Operates without access to the target model's internal parameters or gradients.

Gradient Estimation: Uses a local surrogate model to approximate gradients for discrete text data.

Multi-Constraint Generation: Decodes adversarial text with a joint loss enforcing semantic and grammatical integrity.

High Transferability: Successfully attacks powerful, publicly-available LLMs via transfer.

📚 Citation
If you use TextJosher in your research, please cite our paper:

@article{wang2025textjosher,
  title={TextJosher: A transfer-based black-box attack method Against text classifiers},
  author={Wang, Peishuai and Zhang, Sicong and Xu, Yang and He, Xinlong and Xu, Weida},
  journal={Information Sciences},
  pages={122888},
  year={2025},
  publisher={Elsevier}
}

🔗 Links

Paper: https://doi.org/10.1016/j.ins.2025.122888
