# DER-SSL: Dark Experience Replay with Self-Supervised Learning
*Abstract: Although many continual learning approaches claim to be state-of-the-art, they often do it by defining their own specific setting/evaluation. In this work, we tackle the **class incremental learning setting**, the most difficult and general continual learning setting. We start with Dark Experience Replay (DER), a simple and strong baseline. We extend it to do Self-Supervised Learning to mitigate a common problem in Class-IL known as Prior Information Loss (PIL). We plan to submit our approach to a competition instantiated in Sequoia, a framework which organizes the continual learning research problems to better compare methods with each other.*

DER-SSL will be submitted to supervised learning track of the [CVPR21 Continual Learning Challenge](https://eval.ai/web/challenges/challenge-page/829/overview). 
![Synbols](https://raw.githubusercontent.com/ElementAI/synbols/master/title.png "Synbols Dataset Samples")

## Setup
Ensure you have conda installed. 
```
./install.sh
```

## Training
To run training: 
```
make sl
```

## Refs
* Dark Experience for General Continual Learning: a Strong, Simple Baseline [[paper](https://arxiv.org/abs/2004.07211)][[code](https://github.com/aimagelab/mammoth)]
* Self-Supervised Learning Aided Class-Incremental Lifelong Learning [[paper](https://arxiv.org/abs/2006.05882)]
* Continuous Learning of Context-dependent Processing in Neural Networks (OWM) [[paper](https://arxiv.org/abs/1810.01256)][[code](https://github.com/beijixiong3510/OWM)]
* Synbols: Probing learning algorithmswith synthetic datasets [[paper](https://arxiv.org/abs/2009.06415)]
