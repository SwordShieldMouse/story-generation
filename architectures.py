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
    
