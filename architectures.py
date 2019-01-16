from includes import *

# generate embedding
# instead of using pre-trained embedding, try using an embedding specific to the novel to capture Austen's nuances with less noise?
# or just implement bidirectional transformer for learning
# do BERT style training from https://arxiv.org/pdf/1810.04805.pdf
class Encoder(nn.Module):
    # a stack of N layers for the encoder block
    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        self.N = N
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class SublayerConnection(nn.Module):
    # for applying dropout, residual connection, and then normalizing for any sublayer
    def __init__(self, size, p_dropout = 0.1):
        super(SublayerConnection, self).__init__()

        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

class FeedForward(nn.Module):
    # might as well make this a class since we have to use this for the encoder and decoder
    def __init__(self, d_model, d_ff, p_dropout = 0.1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_eff)
        self.l2 = nn.Linear(d_eff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.relu(self.l1(x))
        return self.l2(self.dropout(x))

class EncoderLayer(nn.Module):
    # a single encoder layer
    def __init__(self, size, self_attn, feedforward, p_dropout = 0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.dropout = nn.Dropout(p_dropout)
        self.sublayer_connections = nn.ModuleList([copy.deepcopy(SublayerConnection(size)) for _ in range(2)])
        self.feedforward = feedforward

    def forward(self, x):
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer_connections[1](x, feedforward)

class Decoder(nn.Module):
    # N-layer decoder with masking for BERT training
    def __init__(self, layer, N):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    # a single decoder layer
    def __init__(self, size, self_attn, src_attn, feedforward, p_dropout = 0.1):
        super(DecoderLayer, self).__init__()

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feedforward = feedforward
        self.sublayers = nn.ModuleList([copy.deepycopy(SublayerConnection(size)) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayers[2](x, feedforward)

def attention(query, key, value, mask = None, dropout = None):
    # compute the scaled dot product attention from the Vaswani paper
    d_k = query_size(-1) # first dimension is batch size
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        # prevents architecture from looking forward in time in the sequence
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, p_dropout = 0.1):
        # h is number of heads
        # d_k is size of lower dim subspace for query, key, value representation
        super(MultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        # first three are for query, key, value
        # last linear layer is for the total output
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])

        self.attn = None
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            # so that the mask can be broadcast across all the heads
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        # project input query, key, value onto subspaces
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)

        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)
