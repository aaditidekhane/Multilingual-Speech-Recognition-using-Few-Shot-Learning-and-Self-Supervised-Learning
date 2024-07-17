# Multilingual-Speech-Recognition-using-Few-Shot-Learning-and-Self-Supervised-Learning
This project is a part of my internship at IIIT-Bangalore uder the guidance of R.V.Subhramanium Sir

# Introduction
This project aims to develop a Foundation Model for End-to-End Automatic Speech Recognition (E2E ASR) using Few-Shot Learning and Self-Supervised Learning (SSL) frameworks. By leveraging the Meta audio dataset, we aim to create a robust and efficient multilingual speech recognition system that can adapt to new languages with minimal labeled data.

# Features
Self-Supervised Learning (SSL)
Self-Supervised Learning is a type of unsupervised learning where the model is trained to predict part of its input from other parts. This project uses SSL to pre-train the model on a large amount of unlabeled multilingual audio data. This allows the model to learn useful audio representations without requiring extensive labeled data, which is often difficult and expensive to obtain.

# Few-Shot Learning (FSL)
Few-Shot Learning aims to enable the model to generalize to new tasks using only a few training examples. In this project, we employ the Matching Networks algorithm, a Meta-/Metric-Learning based Few-Shot Learning algorithm, to enhance the model's capability to recognize speech from languages with only a few labeled examples.

# Matching Networks
Matching Networks are a type of neural network designed for Few-Shot Learning. They use an attention mechanism over a learned embedding of the support set (a small set of labeled examples) to predict the labels of the query set (unlabeled examples). This project extends Matching Networks to handle unsupervised data, making it suitable for a wide range of languages without requiring extensive labeled datasets.

# Project Structure
Data Loading and Preprocessing
Data Collection: The Meta audio dataset is used, which includes diverse multilingual speech data.
Preprocessing: Audio files are converted into Mel spectrograms, which are then normalized and prepared for input into the neural network.
Model Architecture
Self-Supervised Learning Model: A convolutional neural network (CNN) is used to extract meaningful features from the audio data. This model is pre-trained using SSL techniques.
Matching Networks: After pre-training, the model is fine-tuned using the Matching Networks algorithm to handle Few-Shot Learning tasks.
Training and Evaluation
Training: The SSL model is trained on the Meta audio dataset using a reconstruction loss. Subsequently, the Matching Networks are trained to map support set examples to query set examples.

# Evaluation:
The model's performance is evaluated by its accuracy in recognizing speech from the query set using a few examples from the support set.

# Prerequisites
Python 3.7 or higher
PyTorch
Torchaudio
Librosa
Scikit-learn

# Setup
1.Clone this repository.

2.Install the required dependencies using pip install -r requirements.txt.

3.Download the Meta audio dataset using Torchaudio.

4.Running the Model

5. the audio data to generate Mel spectrograms.
6. 
6.Train the Self-Supervised Learning model.

7.Fine-tune the model using Matching Networks for Few-Shot Learning.

8.Evaluate the model's performance on new speech recognition tasks.


# Results
The model achieves competitive accuracy in multilingual speech recognition tasks, demonstrating its ability to generalize to new languages with minimal labeled data. This approach significantly reduces the need for extensive labeled datasets, making it highly scalable and adaptable.

# References
Meta Audio Dataset: Torchaudio Meta Audio Dataset
Self-Supervised Learning: A Simple Framework for Contrastive Learning of Visual Representations
Few-Shot Learning: Matching Networks for One Shot Learning
Mel Spectrograms: Librosa Documentation
PyTorch: PyTorch Documentation
Torchaudio: Torchaudio Documentation

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgements
This project was inspired by recent advancements in Self-Supervised Learning and Few-Shot Learning. Special thanks to R.V Subhramanium (IIIT-B) the authors of the referenced papers and the open-source community for their valuable contributions.
