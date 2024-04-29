from sklearn.model_selection import train_test_split
import json
import pandas as pd
import tensorflow_datasets as tfds

def load_dataset(file_path):
    """
    Load and return the dataset from a JSON file.

    Parameters:
        file_path (str): The file path to the JSON dataset.

    Returns:
        list: List of dataset items.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

dataset_path = '/Users/Hamza/Documents/chatbot/datasets/train.json'
dataset = load_dataset(dataset_path)

def preprocess_data(dataset):
    """
    Extract questions and answers from the dataset, and create a DataFrame.

    Parameters:
        dataset (list): List of dataset items.

    Returns:
        DataFrame: DataFrame containing the questions and answers.
    """
    questions = [item['question'][:100] for item in dataset]
    answers = [item['answer'][:300] for item in dataset]
    return pd.DataFrame({
        'question': questions,
        'answer': answers
    })

data = preprocess_data(dataset)


# Split data into training and validation sets
train, validation = train_test_split(data, test_size=0.2, random_state=4)

# Display the first few rows of the DataFrame
data.head()

def build_vocabulary(text_series):
    """
    Create a vocabulary set from the given pandas Series of text data.

    Parameters:
        text_series (Series): Pandas Series containing text data.

    Returns:
        set: Set of unique words.
    """
    return set(" ".join(text_series.values).split())

vocab_answer = build_vocabulary(train['answer'])
vocab_question = build_vocabulary(train['question'])
vocab_size_ans = len(vocab_answer)
vocab_size_ques = len(vocab_question)
print(f"Vocabulary sizes - Answers: {vocab_size_ans}, Questions: {vocab_size_ques}")

def create_tokenizer(text_series, vocab_size=2**15):
    """
    Create a tokenizer for the given text data using the SubwordTextEncoder.

    Parameters:
        text_series (Series): Text data for tokenizer training.
        vocab_size (int): The target vocabulary size.

    Returns:
        SubwordTextEncoder: A tokenizer configured to the specified vocabulary size.
    """
    return tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        text_series, target_vocab_size=vocab_size)

# tokenizer_ans1 = create_tokenizer(train['answer'])
# tokenizer_ques1 = create_tokenizer(train['question'])

# print(f"Tokenizer vocab sizes - Questions: {tokenizer_ques1.vocab_size}, Answers: {tokenizer_ans1.vocab_size}")
