import torch
from collections import OrderedDict
import transformers
import torch.nn as nn
from copy import deepcopy
from tabtransformer.tabtransformer.tab_transformer_pytorch import CombTabTransformer

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, size, batchnorm=False):
        super().__init__()
        size = list(size)
        size.insert(0, in_dim)
        size.append(out_dim)

        for it, s in enumerate(size):
            if int(s) != s:
                size[it] = int(in_dim * s)

        layers = OrderedDict()

        for it, x1 in enumerate(size[:-1]):
            x2 = size[it + 1]
            layers['linear' + str(it)] = nn.Linear(x1, x2)
            if batchnorm:
                layers['bn' + str(it)] = nn.BatchNorm1d(x2)
            layers['relu' + str(it)] = nn.ReLU(inplace=True)

        self.net = nn.Sequential(layers)

        if torch.cuda.is_available():
            self.net = self.net.cuda()

    def forward(self, x):
        return self.net(x)

class TransformerWithMLP(nn.Module):
    def __init__(self, trns_arch='distilbert', out_dim=1, size=[], batchnorm=False):
        super().__init__()
        if trns_arch == "distilbert":
            self.trns = transformers.AutoModel.from_pretrained('distilbert-base-uncased')
        elif trns_arch == "bert":
            self.trns = transformers.AutoModel.from_pretrained('bert-base-uncased')
        elif trns_arch == "albert":
            self.trns = transformers.AutoModel.from_pretrained('albert-base-v2')
        elif trns_arch == "biobert":
            self.trns = model.AutoModel.from_pretrained(
                    'dmis-lab/biobert-base-cased-v1.1',
                    from_tf=False,
                    config='dmis-lab/biobert-base-cased-v1.1',
                    cache_dir=None
            )
        else:
            raise NotImplementedError("specified transformer architecture not recognized")
        print(torch.cuda.device_count())
        self.trns = self.trns.to('cuda')
        self.trns.train()

        if trns_arch == "distilbert":
            self.in_dim = 768
        else:
            raise NotImplementedError("specify dimension for this transformer architecture")

        self.mlp = SimpleMLP(self.in_dim, out_dim, size, batchnorm=batchnorm)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def forward(self, encoded):
        sequence_output = self.trns(
               encoded["input_ids"],
               attention_mask=encoded["attention_mask"])["last_hidden_state"]
        self.last_hidden_output = sequence_output[:,0,:].view(-1,self.in_dim)

        return self.mlp(sequence_output[:,0,:].view(-1,self.in_dim))

class WithFinetuneLayers(nn.Module):
    def __init__(self, model, current_size, output_size=2, size=[], batchnorm=False):
        super().__init__()
        self.model = deepcopy(model)

        self.finetune = SimpleMLP(current_size, output_size, size, batchnorm=batchnorm) 

    def forward(self, inn):
        out = self.model(inn)

        return self.finetune(out)

class LEvalLayers(nn.Module):
    def __init__(self, current_size, output_size=2, size=[], batchnorm=False):
        super().__init__()
        self.finetune = SimpleMLP(current_size, output_size, size, batchnorm=batchnorm) 

    def forward(self, inn):
        return self.finetune(inn)


class Branch(nn.Module):
    def __init__(self, encoder=None, size=None, repr_dim=64, proj_out=-1):
        super().__init__()

        if not isinstance(encoder, str):
            self.encoder = encoder
        elif encoder == "dnn":
            self.encoder = SimpleMLP(size)
        elif encoder == "resnet":
            if size == 18:
                self.encoder = torchvision.models.resnet18(pretrained=False)
            elif size == 50:
                self.encoder = torchvision.models.resnet50(pretrained=False)
            elif size == 101:
                self.encoder = torchvision.models.resnet101(pretrained=False)
        # TODO NLP option

        if proj_out == -1:
            self.projector = nn.Identity()
        else:
            self.projector = SimpleMLP(repr_dim, proj_out, [(proj_out + repr_dim) / 2], batchnorm=True)

        self.net = nn.Sequential(
            self.encoder,
            self.projector
        )

    def forward(self, x):
        return self.net(x)

class ProjectedTabTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, project_size, batchnorm, inter_dim=32, depth=1, heads=6, attn_dropout=0.1, ff_dropout=0.1, inter_attn=False):
        super().__init__()

        self.mlp = SimpleMLP(in_dim, project_size, size=[], batchnorm=batchnorm)
        self.trns = CombTabTransformer(categories=(), num_continuous=project_size, dim_out=out_dim, dim=32, depth=1, heads=6, attn_dropout=0.1, ff_dropout=0.1, inter_attn=False, embedding_type='linear')

    def forward(self, x):
        out = self.mlp(x)
        return self.trns({"categorical": None, "continuous": out})

def Pathological_Transformer(trns_arch='distilbert', out_dim=1, size=[], batchnorm=True, classification=False):
    return TransformerWithMLP(trns_arch=trns_arch, out_dim=out_dim, size=size, batchnorm=batchnorm)
    
def Genetic_MLP(in_dim, out_dim, size=[1280], batchnorm=True):
    return SimpleMLP(in_dim, out_dim, size=size, batchnorm=batchnorm) 

def Genetic_TabTransformer(in_dim, out_dim, project_size=560, batchnorm=True):
    return ProjectedTabTransformer(in_dim, out_dim, project_size, batchnorm, inter_dim=32, depth=1, heads=6, attn_dropout=0.1, ff_dropout=0.1, inter_attn=False)

def Clinical_TabTransformer(out_dim=1, categories=-1, num_cont=-1):
    return CombTabTransformer(categories=categories, num_continuous=num_cont, dim_out=out_dim, dim=32, depth=1, heads=6, attn_dropout=0.1, ff_dropout=0.1, inter_attn=False, embedding_type='linear').cuda()

def TCGAEncoders(data_types=['rna-seq'], datahandler=None, mode="validate", rep_dim=None, rna_hidden=[], gpu=None, trns_arch=trns_arch):
    if datahandler == None:
        raise ValueError("TCGA Encoder generation method must receive a dataset!")
    if gpu != None:
        torch.cuda.set_device(gpu)
    if mode == "validate":
        task_is_classification = True

        if data_types[0] == "rna-seq":
            input_dim = datahandler[data_types[0]][0][data_types[0]]
            input_dim = input_dim.size()[0]

            assert(len(data_types) == 1)
            return [Genetic_MLP(input_dim, rep_dim)]
        elif data_types[0] in ["reports", "clean-reports"]:
            return [Pathological_Transformer(out_dim=rep_dim)]
        elif data_types[0] == "toy_reports":
            return [Pathological_Transformer(out_dim=rep_dim)]
        elif data_types[0] == "clinical":
            categorical = dset["categorical"]
            continuous = dset["continuous"]

            num_classes = tuple(map(int, torch.max(categorical + 1, dim=0)[0].tolist()))
            num_cont = continuous.size()[1]

            return [Clinical_TabTransformer(out_dim=rep_dim, categories=num_classes, num_cont=num_cont)]
        else:
            raise NotImplementedError("only rna seq has been implemented for model generation")
    else:
        def encoder_map(dtype, dset):
            if dtype == "rna-seq":
                input_dim = dset.size()[1]
                return Genetic_MLP(input_dim, rep_dim, size=rna_hidden)
                #return Genetic_TabTransformer(input_dim, rep_dim)
            elif dtype in ["reports", "clean-reports"]:
                return Pathological_Transformer(out_dim=rep_dim, trns_arch=trns_arch)
            elif dtype == "toy_reports":
                return Pathological_Transformer(out_dim=rep_dim)
            elif dtype == "clinical":
                categorical = dset["categorical"]
                continuous = dset["continuous"]

                num_classes = tuple(map(int, torch.max(categorical + 1, dim=0)[0].tolist()))
                num_cont = continuous.size()[1]

                return Clinical_TabTransformer(out_dim=rep_dim, categories=num_classes, num_cont=num_cont)
            else:
                raise ValueError("don't recognize encoder specificaion type")

        encoders = []

        for dtype in data_types:
            encoders.append(encoder_map(dtype, datahandler.pretrain[:][dtype])) 

            if gpu != None:
                encoders[-1].to(gpu)

        return encoders
