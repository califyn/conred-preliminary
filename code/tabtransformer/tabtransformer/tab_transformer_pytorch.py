import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class InterSampleAttention(nn.Module):
    def __init__(self, num_pts, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        self.attn = Attention(dim * num_pts, heads=heads, dim_head=dim_head, dropout=dropout)

    def forward(self, x):
        orig_size = x.size()
        x = x.reshape((1, x.size()[0], -1))
        return self.attn(x).reshape(orig_size)

class MixedEmbedding(nn.Module):
    def __init__(self, num_tokens, dim, num_cont, embedding='linear'):
        super().__init__()
        self.embedding = embedding
        self.embeds = nn.Embedding(num_tokens, dim)
        if num_cont > 0:
            self.num_cont = num_cont
            self.dim = dim
            if embedding == 'linear':
                self.cont_embed_low = nn.Embedding(num_cont, dim)
                self.cont_embed_high = nn.Embedding(num_cont, dim)
            elif embedding == 'nn':
                for it in range(num_cont):
                    setattr(self, 'cont_embed-' + str(it), 
                        nn.Sequential(
                            nn.Linear(1, dim // 4),
                            nn.ReLU(),
                            nn.Linear(dim // 4, dim // 2),
                            nn.ReLU(),
                            nn.Linear(dim // 2, dim)
                        )
                    )
            else:
                raise ValueError("embedding method either linear or nn")

    def forward(self, x, cont=None):
        if x != None:
            x = self.embeds(x)
        if cont != None:
            assert(self.num_cont == cont.size()[1])
            if self.embedding == 'linear':
                range_idx = torch.arange(self.num_cont).cuda()
                c = self.cont_embed_low(range_idx) + torch.unsqueeze(self.cont_embed_high(range_idx), 0) * torch.unsqueeze(cont, 2)
            else:
                c = torch.cat([torch.unsqueeze(getattr(self, 'cont_embed-' + str(it))(cont[:, it].reshape((-1, 1))), dim=1) for it in range(self.num_cont)], dim=1)
            if x != None:
                x = torch.cat((x, c), dim=1)
            else:
                x = c

        return x


# transformer

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout, inter_attn=False):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if not inter_attn:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    Residual(PreNorm(dim, InterSampleAttention(int(num_cats + num_cont), dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
                ]))

    def forward(self, x):
        x = self.embeds(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x

class CombTransformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout, num_cont=0, inter_attn = False, num_cats=None, embedding_type='linear'):
        super().__init__()
        self.embed = MixedEmbedding(num_tokens, dim, num_cont, embedding=embedding_type)

        self.layers = nn.ModuleList([])

        self.inter_attn = inter_attn
        for _ in range(depth):
            if not inter_attn:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    Residual(PreNorm(dim, InterSampleAttention(int(num_cats + num_cont), dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
                ]))

    def forward(self, x, cont=None):
        x = self.embed(x, cont)

        if self.inter_attn:
            for attn, i_attn, ff in self.layers:
                x = attn(x)
                x = i_attn(x)
                x = ff(x)
        else:
            for attn, ff in self.layers:
                x = attn(x)
                x = ff(x)

        return x
# mlp

class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# main class

class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 2,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        inter_attn = False
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # continuous

        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        # transformer

        self.transformer = Transformer(
            num_tokens = total_tokens,
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            inter_attn = inter_attn
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)

    #def forward(self, x_categ, x_cont):
    def forward(self, x):
        x_categ = x["categorical"]
        x_cont = x["continuous"]

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        x_categ += self.categories_offset

        x = self.transformer(x_categ)

        flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim = -1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        x = torch.cat((flat_categ, normed_cont), dim = -1)
        return self.mlp(x)

class CombTabTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 2,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        inter_attn = False,
        embedding_type = 'linear'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # continuous

        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        # transformer

        self.transformer = CombTransformer(
            num_tokens = total_tokens,
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            num_cont = num_continuous,
            inter_attn = inter_attn,
            num_cats = self.num_categories,
            embedding_type = embedding_type
        )

        # mlp to logits

        #input_size = (dim * self.num_categories) + num_continuous
        input_size = dim * (self.num_categories + num_continuous)
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)

    #def forward(self, x_categ, x_cont):
    def forward(self, x):
        x_categ = x["categorical"]
        x_cont = x["continuous"]

        if x["categorical"] == None and self.num_continuous == 0:
            assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
            x_categ += self.categories_offset

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim = -1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        x = self.transformer(x_categ, cont=x_cont)

        flat_out = x.flatten(1)

        return self.mlp(flat_out)

