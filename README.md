# 696_predict_text_difficulity
#### Predict text difficulty in Milestone 2 of SIADS 696
The project aims to predict the text difficulty based on the data provided by the project guideline. This repo is consisted of 4 parts with its file/folder name:
- Dataset:  01_data
- Unsupervised learning to explore the features or predict text diffculity:  02_Text_Clustering.ipynb
- Supervised lerarning to predict text diffuclity in 2 approaches: 
  - Bert based embedding + classification: 03_Text classification with BERT in PyTorch.ipynb
  - tf-idf + logstic regression: 04_Traditional text classification with Scikit-learn.ipynb
- zero shot / few shots to supplement supervised learning to improve the performance: 05_Zero-Shot Text Classification.ipynb

For unsupervised learning, the idea is:
  - As the 1st approach, tokenizing texts by BERT, and comparing the text in difficulty with the text not in difficulty via cosine similarity, then try to find out the features of both types, and try to predict the text difficulty.
  - As the 2nd approach, through bag of words / tf-idf, try to find out the differrences of words between text in difficulty and text not in difficulty to try to predict the text difficulty.
  - Furthermore, based on the features detected in the above, those could be some input for the zero/few shots to try to improve the performance of supervised learning.

For supervised learning, it focuses on comparing the performance results between tradional text classification and BERT-based text classification to see the better one to predict the text diffculty. On top of this, zero / few shot will be considered to be applied to improve the performance of prediction.
