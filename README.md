# Next Word Prediction Using LSTM

## Project Overview

This project implements a next word prediction system using Long Short-Term Memory (LSTM) neural networks. The model learns from a custom dataset containing frequently asked questions and answers about a web development bootcamp, and then predicts the next word in a given sequence of text.

## What Problem Does This Solve?

In natural language processing, predicting the next word in a sequence is a fundamental task with applications in autocomplete features, text generation, and language modeling. This project demonstrates how recurrent neural networks, specifically LSTMs, can learn patterns in text data and make contextually relevant predictions.

## Technical Implementation

### Dataset

The project uses a custom text corpus containing FAQ content about a Full Stack Web Development Bootcamp. The dataset includes information about:
- Course fees and payment structure (Rs 999/month subscription)
- Syllabus covering MERN stack (HTML, CSS, JavaScript, React, Node.js, MongoDB, etc.)
- Course policies (refunds, recordings, access duration)
- Certification requirements and placement assistance details

The choice of this specific dataset allows the model to learn domain-specific vocabulary and sentence structures commonly found in educational content.

### Architecture Details

**Model Structure:**
- Embedding Layer: 251 vocabulary size, 100-dimensional embeddings, input length of 19
- LSTM Layer: 150 hidden units
- Dense Output Layer: 251 units with softmax activation

**Key Parameters:**
- Vocabulary Size: 251 unique words
- Maximum Sequence Length: 20 words (after padding)
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Training Epochs: 100

### Data Preprocessing Pipeline

1. **Tokenization**: Used TensorFlow's Tokenizer to convert text into numerical sequences
2. **N-gram Generation**: Created training sequences by generating all possible n-grams from each sentence
3. **Padding**: Applied pre-padding to standardize sequence lengths to maximum length of 20
4. **Label Encoding**: Converted target words to one-hot encoded vectors (categorical format)
5. **Train-Test Split**: Features (X) contain all words except the last, target (y) contains the predicted word

### How the Prediction Works

The model takes an input phrase and:
1. Tokenizes the input text into numerical sequences
2. Pads the sequence to match the training format
3. Passes it through the trained LSTM network
4. Uses argmax on softmax output to select the most probable next word
5. Appends the predicted word and repeats for sequential predictions

The inference loop demonstrates iterative prediction by generating 10 consecutive words starting from the seed text "We cover".

## Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for building and training the LSTM model
- **NumPy**: Numerical computations and array operations
- **Tokenizer**: Text preprocessing and tokenization

## Key Learning Outcomes

Through this project, I gained practical experience in:
- Implementing recurrent neural networks for sequence modeling
- Preprocessing text data for deep learning applications
- Understanding LSTM architecture and its advantages over vanilla RNNs
- Working with embedding layers to represent words as dense vectors
- Handling variable-length sequences using padding techniques
- Training neural networks and monitoring convergence

## Challenges Faced

1. **Sequence Length Variability**: Different sentences had varying lengths, which required careful padding strategy to maintain information while standardizing inputs
2. **Small Dataset Size**: Working with a limited corpus meant the model's vocabulary and generalization were constrained
3. **Training Time**: Running 100 epochs required balancing training time with model accuracy

## Potential Improvements

- Expand the dataset to include more diverse text for better generalization
- Implement bidirectional LSTM for better context understanding
- Add dropout layers to prevent overfitting
- Experiment with different embedding dimensions and LSTM units
- Implement beam search for more diverse predictions
- Add model checkpointing to save the best weights during training
- Create a proper train-test split for validation

## How to Run

1. Install required dependencies:
   ```
   pip install tensorflow numpy
   ```

2. Run the Python script:
   ```
   python lstm_nextword_prediction.py
   ```

   Or open and execute the Jupyter notebook:
   ```
   jupyter notebook LSTM_NextWord_Prediction.ipynb
   ```

3. The model will train for 100 epochs and then generate 10 predicted words starting from "We cover"

## Results

The model successfully learns patterns in the training data and can predict contextually relevant next words. Starting with "We cover", the model predicts subsequent words based on the learned patterns from the bootcamp syllabus content.

## Project Structure

```
Next_Word_Prediction_LSTM/
├── lstm_nextword_prediction.py      # Main Python implementation
├── LSTM_NextWord_Prediction.ipynb   # Jupyter notebook version
├── README.md                         # Project documentation
└── LICENSE                           # License file
```

## Future Scope

This project serves as a foundation for more advanced NLP applications such as:
- Chatbot development
- Text autocomplete systems
- Content generation tools
- Language translation systems
- Sentiment analysis enhancement
