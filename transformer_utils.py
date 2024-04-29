
import tensorflow as tf
import numpy as np

def get_angles(pos, i, d_model):
    """
    Generate angle values for positional encoding.

    Parameters:
        pos (array): Positions.
        i (array): Dimension indices.
        d_model (int): Dimensionality of the model.

    Returns:
        array: Angle values.
    """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """
    Compute the positional encoding matrix.

    Parameters:
        position (int): Maximum position encoding length.
        d_model (int): Model dimensionality.

    Returns:
        tensor: The positional encoding matrix.
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, :]
    return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(151, 512)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate the scaled dot-product attention.

    Parameters:
        q (tf.Tensor): Query tensor of shape (..., seq_len_q, depth).
        k (tf.Tensor): Key tensor of shape (..., seq_len_k, depth).
        v (tf.Tensor): Value tensor of shape (..., seq_len_v, depth_v).
        mask (tf.Tensor, optional): Float tensor that can be broadcast to 
                                    (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        tuple: (output, attention_weights) where 'output' is the attention output and
               'attention_weights' are the weights computed by the attention mechanism.
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # Dot product of query and key
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # Depth of key tensor for scaling
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # Apply the mask by adding a large negative number to masked positions

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # Softmax over the last axis (seq_len_k)
    output = tf.matmul(attention_weights, v)  # Weighted sum of values based on computed attention weights

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer for a Transformer model.

    Attributes:
        num_heads (int): Number of attention heads.
        d_model (int): Total dimension of the model.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension of tensor 'x' into (num_heads, depth) and transpose the result.
        
        Parameters:
            x (tf.Tensor): Tensor to split.
            batch_size (int): Batch size of input tensor.
        
        Returns:
            tf.Tensor: Transposed tensor with shape (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # Linear transformation
        k = self.wk(k)  # Linear transformation
        v = self.wv(v)  # Linear transformation

        q = self.split_heads(q, batch_size)  # Split into heads
        k = self.split_heads(k, batch_size)  # Split into heads
        v = self.split_heads(v, batch_size)  # Split into heads

        # Perform scaled dot product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # Transpose and reshape the output to consolidate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear transformation
        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """
    Create a pointwise feed forward network consisting of two dense layers.

    Parameters:
        d_model (int): Dimensionality of the input tensor.
        dff (int): Dimensionality of the hidden layer.

    Returns:
        tf.keras.Sequential: A sequential model representing the feed forward network.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(d_model)  # Output layer
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer for a Transformer model, comprising multi-head attention and feed forward network.

    Parameters:
        d_model (int): Dimensionality of the layer output.
        num_heads (int): Number of attention heads.
        dff (int): Dimensionality of the feed forward network's hidden layer.
        rate (float): Dropout rate.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # Multi-head attention layer
        attn_output, _ = self.mha(x, x, x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class Encoder(tf.keras.layers.Layer):
    """
    Encoder component of a Transformer model.

    Parameters:
        num_layers (int): Number of sub-encoder-layers in the encoder.
        d_model (int): Dimensionality of the encoder layers and output.
        num_heads (int): Number of attention heads.
        dff (int): Dimensionality of the feed forward network's hidden layer.
        input_vocab_size (int): Size of the input vocabulary.
        maximum_position_encoding (int): The size of the position encoding.
        rate (float): Dropout rate.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x  # The final output tensor of the encoder

class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder layer for a Transformer model, comprising two multi-head attention mechanisms and a feed forward network.

    Parameters:
        d_model (int): Dimensionality of the layer output.
        num_heads (int): Number of attention heads.
        dff (int): Dimensionality of the feed forward network's hidden layer.
        rate (float): Dropout rate.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # First multi-head attention layer
        attn1, attn_weights_block1 = self.mha1(x, x, x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Second multi-head attention layer
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Feed forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
    """
    Decoder class for a Transformer model comprising multiple decoder layers.

    Attributes:
        num_layers (int): Number of decoder layers.
        d_model (int): Dimensionality of the output space for each layer.
        num_heads (int): Number of attention heads.
        dff (int): Dimension of the feed-forward network.
        target_vocab_size (int): Size of the target vocabulary.
        maximum_position_encoding (int): Maximum length of the position encoding.
        rate (float): Dropout rate used in decoder layers.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate scheduler with a warm-up phase.

    Attributes:
        d_model (int): The base dimensionality of the layers.
        warmup_steps (int): Number of steps to linearly increase the learning rate.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    """
    Sparse categorical crossentropy loss function for training.

    Parameters:
        real (tf.Tensor): The actual labels.
        pred (tf.Tensor): The predictions returned by the model.

    Returns:
        tf.Tensor: Computed loss value.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
