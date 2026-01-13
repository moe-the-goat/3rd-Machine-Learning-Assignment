OTHER MODELS TRIED
==================

This folder contains alternative models that were experimented with during 
the project development but were not selected as the final models.

FOLDER STRUCTURE:
-----------------
Other_Models_Tried/
├── code/                    # Python scripts for each model
│   ├── svm_model.py
│   ├── logreg_model.py
│   └── textcnn_model.py
├── results/                 # Output results from each model
│   ├── SVM/
│   ├── LogisticRegression/
│   └── TextCNN/
└── README.txt

FINAL CHOSEN MODELS (Task 4):
- Random Forest (traditional ML)
- Transformer/DistilBERT (deep learning)

OTHER MODELS:
-------------

1. SVM (Support Vector Machine)
   - Performance: ~83% accuracy, 81% Macro F1
   - Similar to Random Forest but slightly slower to train
   - Good linear separability in TF-IDF space

2. TextCNN (Convolutional Neural Network)
   - Performance: ~81% accuracy with GloVe embeddings
   - Required pretrained embeddings (GloVe) to work well
   - More complex to tune than traditional methods
   - Lower performance than Transformer

HOW TO RUN:
-----------
cd Other_Models_Tried/code
python svm_model.py
python textcnn_model.py

REASONING FOR FINAL MODEL SELECTION:
------------------------------------
- Random Forest: Best traditional ML model, interpretable, fast
- Transformer: Best overall performance (87% acc), leverages pretrained knowledge

These alternative models provide useful comparison points but the chosen 
models (RF + Transformer) offer the best balance of performance and practicality.
