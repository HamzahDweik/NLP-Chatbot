from transformer_utils import *

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pe_input = pe_input
        self.pe_target = pe_target
        self.rate = rate
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
        if enc_padding_mask is None or look_ahead_mask is None or dec_padding_mask is None:
            enc_padding_mask, look_ahead_mask, dec_padding_mask = Transformer.create_masks(inp, tar)
        
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    def get_config(self):
        return {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_vocab_size": self.input_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "pe_input": self.pe_input,
            "pe_target": self.pe_target,
            "rate": self.rate
        }
    
    @staticmethod
    def create_padding_mask(seq):
        """
        Create a padding mask for batches of sequences.

        Parameters:
            seq (array): Batch of sequences.

        Returns:
            tensor: A padding mask tensor.
        """
        # Mask all the pad tokens in the batch of sequence to ensure the model does not treat padding as the input.
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # Add new axes to match the shape required for the attention mechanism.
        return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        """
        Create a look-ahead mask to mask future tokens in a sequence during training.
        
        This mask helps prevent the model from peeking at future tokens when making predictions.

        Parameters:
            size (int): Size of the mask (typically the sequence length).

        Returns:
            tf.Tensor: A 2D mask tensor with dimensions [size, size].
        """
        # Create a lower triangular matrix using `tf.linalg.band_part`
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # Returns a tensor where the upper triangle is masked with True (1s)
    
    @staticmethod
    def create_masks(inp, tar):
        """
        Create all masks required for training/validation of the Transformer model.

        Parameters:
            inp (tf.Tensor): Input tensor.
            tar (tf.Tensor): Target tensor.

        Returns:
            tuple: Encoder padding mask, combined mask for decoder, decoder padding mask.
        """
        enc_padding_mask = Transformer.create_padding_mask(inp)
        dec_padding_mask = Transformer.create_padding_mask(inp)
        look_ahead_mask = Transformer.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = Transformer.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask