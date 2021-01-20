import torch
import math, copy


def clone_layers(module, N):
    """
    Produce N identical layers.
    """
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask = None, dropout = None):
    """
    Compute 'Scaled Dot Product Attention'

    Notes:
    ------
    vectors are treated as row vectors.
    """
    d_k = query.size(-1)
    dot_prod = torch.matmul(query, key.transpose(-2, -1))/(math.sqrt(d_k))
    if mask is not None:
        dot_prod = dot_prod.masked_fill(mask == 0, -1e9)
    p_attn = torch.nn.functional.softmax(dot_prod, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn) # sets some attentions to 0 
    attn = torch.matmul(p_attn, value)
    return attn, p_attn


class MultiHeadedAttention(torch.nn.Module):
    """
    We assume d_v always equals d_k.
    """
    def __init__(self, num_heads, d_model, dropout = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_W_O = torch.nn.Linear(d_model, d_model, bias = False)
        # projection for all heads simultaniously => dim is h * d_k = d_model
        self.linear_W_Q = torch.nn.Linear(d_model, d_model, bias = False) 
        self.linear_W_K = torch.nn.Linear(d_model, d_model, bias = False)
        self.linear_W_V = torch.nn.Linear(d_model, d_model, bias = False)

        self.linears = torch.nn.ModuleList([self.linear_W_Q, self.linear_W_K, self.linear_W_V, self.linear_W_O])

        self.p_attn = None
        self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        """
        MultiHead(Q, K, V) = (concat(h_1, h_2, ..., h_n))W^O 
        where
        h_i = attention(QW_i^Q, KW_i^K, VW_i^V)
        """
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0) # nbatches x d_model

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1,2)
                            for l, x in zip(self.linears[:-1], (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.p_attn = attention(query, key, value, mask = mask, 
                                    dropout = self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


class LayerNorm(torch.nn.Module):
    """
    Construct a layer normalization from https://arxiv.org/pdf/1607.06450.pdf
    """
    def __init__(self, features_shape, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features_shape))
        self.b_2 = torch.nn.Parameter(torch.zeros(features_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * ((x - mean) / (std + self.eps)) + self.b_2


class PossitionwiseFeedForward(torch.nn.Module):
    """
    Implements Feed-forward from https://arxiv.org/abs/1706.03762

    max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PossitionwiseFeedForward, self).__init__()
        self.proj_1 = torch.nn.Linear(d_model, d_ff, bias = True)
        self.proj_2 = torch.nn.Linear(d_ff, d_model, bias = True)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.proj_2(self.dropout(torch.nn.functional.relu(self.proj_1(x))))


class EncoderLayer(torch.nn.Module):
    """
    """
    def __init__(self, num_heads, d_model, d_ff, dropouts):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, d_model, dropouts['self_attn'])
        self.dropout_attn = torch.nn.Dropout(dropouts['self_attn_skip'])
        self.norm_attn = LayerNorm(d_model)
        
        self.feed_forward = PossitionwiseFeedForward(d_model, d_ff, dropouts['ff'])
        self.dropout_ff = torch.nn.Dropout(dropouts['ff_skip'])
        self.norm_ff = LayerNorm(d_model)

        self.size = d_model

    def forward(self, x, mask):
        x = self.norm_attn(x + self.dropout_attn(self.self_attn(x, x, x, mask))) # LayerNorm(x + dropout(sublayer(x)))
        return self.norm_ff(x + self.dropout_ff(self.feed_forward(x))) # LayerNorm(x + dropout(sublayer(x)))


class Encoder(torch.nn.Module):
    """
    Core encoder is a stack of N layers.
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone_layers(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(torch.nn.Module):
    """
    Deoder layer is made from self-attn, source-attn, and feed-forward layers
    """
    def __init__(self, num_heads, d_model, d_ff, dropouts):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, d_model, dropouts['self_attn'])
        self.dropout_self_attn = torch.nn.Dropout(dropouts['self_attn_skip'])
        self.norm_self_attn = LayerNorm(d_model)

        self.src_attn = MultiHeadedAttention(num_heads, d_model, dropouts['src_attn'])
        self.dropout_src_attn = torch.nn.Dropout(dropouts['src_attn_skip'])
        self.norm_src_attn = LayerNorm(d_model)

        self.feed_forward = PossitionwiseFeedForward(d_model, d_ff, dropouts['ff'])
        self.dropout_ff = torch.nn.Dropout(dropouts['ff_skip'])
        self.norm_ff = LayerNorm(d_model)

        self.size = d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Assuming that keys and values are the same.
        """
        m = memory
        x = self.norm_self_attn(x + self.dropout_self_attn(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm_src_attn(x + self.dropout_src_attn(self.src_attn(x, m, m, src_mask)))
        return self.norm_ff(x + self.dropout_ff(self.feed_forward(x)))


class Decoder(torch.nn.Module):
    """
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone_layers(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Embeddings(torch.nn.Module):
    """
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = torch.nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

        
class PositionalEncoding(torch.nn.Module):
    """
    PE_(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE_(pos, 2i+1) = cos(pos/10000^(2i/d_model))

    Encoding is added.
    """
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad_(False)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Generator(torch.nn.Module):
    """
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = torch.nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.nn.functional.log_softmax(self.proj(x), dim = -1)


class EncoderDecoder(torch.nn.Module):
    """
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        # Remember generator for easier transport
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)



def create_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, num_heads=8, dropout = 0.1):
    """
    using same dropout for all layers
    """
    dropouts_encoder = {'self_attn' : dropout,
                        'self_attn_skip' : dropout, 
                        'ff' : dropout,
                        'ff_skip' : dropout}
    dropouts_decoder = {'self_attn' : dropout,
                        'self_attn_skip' : dropout, 
                        'src_attn' : dropout,
                        'src_attn_skip' : dropout,
                        'ff' : dropout,
                        'ff_skip' : dropout}
    dropout_src_embed = dropout
    dropout_tgt_embed = dropout
    model = EncoderDecoder(encoder = Encoder(EncoderLayer(num_heads, d_model, d_ff, dropouts_encoder), N),
                            decoder = Decoder(DecoderLayer(num_heads, d_model, d_ff, dropouts_decoder), N),
                            src_embed = torch.nn.Sequential(Embeddings(d_model, src_vocab), 
                                                            PositionalEncoding(d_model, dropout_src_embed)),
                            tgt_embed = torch.nn.Sequential(Embeddings(d_model, tgt_vocab),
                                                            PositionalEncoding(d_model, dropout_tgt_embed)),
                            generator = Generator(d_model, tgt_vocab))

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model

    