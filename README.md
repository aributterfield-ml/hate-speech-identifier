# Hate Speech Identifier

## Introduction
Hate speech is a growing concern in online communication. While excessive censorship may stifle free speech, counter speech presents an alternative—addressing hateful content directly. This project aims to develop a machine learning model that classifies hate speech, which can potentially be used alongside a counter speech generator.

## Data
The dataset used is **Multitarget_CONAN.csv**, containing over 5,000 pairs of hate speech and counter speech. Each pair was separated and labeled:  
- `1` for hate speech  
- `0` for counter speech  

The dataset was split into **80% training** and **20% testing**, resulting in 10,000 individual entries.

## Methodology
Two models were implemented for classification:  

1. **Naïve Bayes Classifier (TF-IDF)**  
   - Tokenized text and computed **TF-IDF scores**  
   - Trained using a **Multinomial Naïve Bayes classifier**  
   - Provided a baseline for comparison  

2. **Convolutional Neural Network (CNN)**  
   - Three different preprocessing approaches:
     - No preprocessing  
     - Removing punctuation, stop words, and lowercasing  
     - Additional **lemmatization**  
   - Model: **Keras Sequential() CNN** with:
     - **Three dense layers** (256 → 128 → 1)  
     - **ReLU activation** for hidden layers, **Sigmoid** for output  
     - **Adam optimizer** and **log loss function**  
     - **Early stopping** to prevent overfitting  

## Results

| Model | Accuracy |
|--------|----------|
| Naïve Bayes with TF-IDF | 90.11% |
| CNN (no preprocessing) | 91.31% |
| CNN (with stripping & stop words) | 90.26% |
| CNN (with lemmatization) | 90.00% |

## Discussion
- **Preprocessing had minimal impact** on CNN performance.  
- **Lemmatization reduced accuracy**, likely due to unintended word modifications.  
- **CNN outperformed Naïve Bayes**, but the difference was small, suggesting both are viable for hate speech detection.  
- This model, combined with a **counter speech generator**, could effectively **combat hate speech** in public forums.

## Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/hate-speech-detector.git
   cd hate-speech-detector
