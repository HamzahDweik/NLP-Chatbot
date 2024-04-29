import tensorflow as tf
import tensorflow_datasets as tfds
from transformer import *
import pickle

def generate_text_(input_text, model, tokenizer_q, tokenizer_a):
    """
    Generates text from a given input string using the provided Transformer model,
    loading the required tokenizers from specified paths.
    """
    # Process the input text
    start_token = [tokenizer_q.vocab_size]
    end_token = [tokenizer_q.vocab_size + 1]
    inp_sentence = start_token + tokenizer_q.encode(input_text) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_a.vocab_size]
    decoder_input = tf.expand_dims(decoder_input, 0)

    MAX_LENGTH = 1000
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = Transformer.create_masks(encoder_input, decoder_input)
        predictions, attention_weights = model(encoder_input, decoder_input, False,
                                               enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == tokenizer_a.vocab_size + 1:
            return tokenizer_a.decode([token for token in tf.squeeze(decoder_input, axis=0).numpy() if token < tokenizer_a.vocab_size])

        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

    return tokenizer_a.decode([token for token in tf.squeeze(decoder_input, axis=0).numpy() if token < tokenizer_a.vocab_size])

def import_tokenizer(file_prefix):
    """
    Import a SubwordTextEncoder tokenizer from a file.

    Parameters:
        file_prefix (str): The file prefix from which to load the tokenizer. This should include the path and the base file name, but not the file extensions.

    Returns:
        SubwordTextEncoder: The loaded tokenizer.
    """
    return tfds.deprecated.text.SubwordTextEncoder.load_from_file(file_prefix)

def evaluate(inp_sentence, model, tokenizer_q, tokenizer_a):
    """
    Evaluates an input sentence using the provided model and tokenizers, returning the sequence
    of predicted tokens and attention weights.

    Parameters:
        inp_sentence (str): The sentence to evaluate.
        model (tf.keras.Model): The Transformer model.
        tokenizer_q (Tokenizer): Tokenizer for questions.
        tokenizer_a (Tokenizer): Tokenizer for answers.

    Returns:
        tuple: A tuple containing the sequence of predicted tokens and attention weights.
    """
    start_token = [tokenizer_q.vocab_size]
    end_token = [tokenizer_q.vocab_size + 1]
    inp_sentence = start_token + tokenizer_q.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_a.vocab_size]
    decoder_input = tf.expand_dims(decoder_input, 0)
    MAX_LENGTH = 500
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = Transformer.create_masks(encoder_input, decoder_input)
        predictions, attention_weights = model(encoder_input, decoder_input, False,
                                               enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]  # Focus on the last word
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == tokenizer_a.vocab_size + 1:
            return tf.squeeze(decoder_input, axis=0), attention_weights

        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

    return tf.squeeze(decoder_input, axis=0), attention_weights

def reply(sentence, transformer, tokenizer_q, tokenizer_a):
    """
    Generates a reply for a given sentence using the transformer model and can optionally plot attention weights.

    Parameters:
        sentence (str): The input sentence to translate.
        transformer (Model): A transformer model that performs the translation.
        tokenizer_q (Tokenizer): Tokenizer for questions.
        tokenizer_a (Tokenizer): Tokenizer for answers.
        plot (str, optional): If provided, specifies which attention layer's weights to plot.

    Returns:
        tuple: The input sentence and its predicted translation.
    """
    result, attention_weights = evaluate(sentence, transformer, tokenizer_q, tokenizer_a)
    predicted_sentence = tokenizer_a.decode([i for i in result if i < tokenizer_a.vocab_size])  
  
    # print('Input: {}'.format(sentence))
    # print('Predicted translation: {}'.format(predicted_sentence))
    # if plot:
    #     plot_attention_weights(attention_weights, tokenizer_q, tokenizer_a, sentence, result, plot)
    return sentence, predicted_sentence

def load_guru():

    # Load the tokenizers
    with open('saves/tokenizers/tokenizer_ques.pickle', 'rb') as handle: tokenizer_ques = pickle.load(handle)
    with open('saves/tokenizers/tokenizer_ans.pickle', 'rb') as handle: tokenizer_ans = pickle.load(handle)

    # Initialize Transformer with updated parameters
    transformer = Transformer(4, 512, 16, 2048, 33665, 33009, 
                            pe_input=33665, pe_target=33009, rate=0.1)

    # Initialize optimizer with custom learning rate schedule
    learning_rate = CustomSchedule(512)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Set up checkpoint directory
    checkpoint_path = "./checkpoints_test/train151"

    # Create and manage checkpoints
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # Restore the latest checkpoint if it exists
    if ckpt_manager.latest_checkpoint:
        load_status = ckpt.restore(ckpt_manager.latest_checkpoint)
        load_status.expect_partial()  # This suppresses the warnings about unmatched variables
        # print('Latest checkpoint restored!!')
    return transformer, tokenizer_ques, tokenizer_ans
