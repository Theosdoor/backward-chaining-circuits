# A Mechanistic Analysis of a Transformer Trained on Symbolic Multi-Step Reasoning Task

This is the official implementation of [A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task](https://arxiv.org/abs/2402.11917).

<p align="center">
      <img src="https://github.com/abhay-sheshadri/backward-chaining-circuits/assets/62884101/79b7add7-993f-47cb-9187-d10c11b3e331" style="width:400px"/>
</p>
<p align="center">
      <em>Figure 1: Given an input prompt, the model concatenates edge tokens in a single token position (A), and copies the goal node into the final token position (B). Then, the next step is identified by applying an iterative algorithm that climbs the tree one level per layer (C).</em>
</p>  



## Usage

#### 1. Dependencies
To install dependencies:
```setup
conda env update --file environment.yml
```

#### 2. Training and Evaluation Code
To train a model from scratch or continue the training, use `training.py`. We provide functions that have been used for aanay in `src/utils.py`.

#### 3. Pre-trained Model
The model checkpoint we studied in our work is provided in `model.pt`. 

#### 4. Replication of Results
The notebook `figures.ipynb` replicates all figures we report in our paper.

## More info
In our experiments, we use a 6-layer, decoderonly transformer with an embedding dimension
of 128, a single attention head per layer, and a feedforward dimension of 512, resulting in a total of
1.2 million parameters. The training dataset consists of 150,000 generated trees.
The edge lists of
these trees are shuffled to prevent the model from
learning simple heuristics and encourage structural
understanding of trees. To evaluate the performance of our model, we compute the accuracy
based on the exact match of complete sequences
using greedy decoding. Our model achieves 99.7 %
accuracy rate on a test set of 15,000 unseen trees,
despite seeing just a small fraction of all possible
trees during training (see Appendix B). This suggests that generalization is required for meaningful
performance and that the model has learned to be
capable of solving pathfinding in trees.



## Citation Information
BibTeX citation:
```bibtex
@misc{brinkmann2024mechanistic,
      title={A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task}, 
      author={Jannik Brinkmann and Abhay Sheshadri and Victor Levoso and Paul Swoboda and Christian Bartelt},
      year={2024},
      eprint={2402.11917},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
