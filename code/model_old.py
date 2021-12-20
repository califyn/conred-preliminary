import torch
from collections import OrderedDict
import transformers
import torch.nn as nn

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

    def forward(self, encoded):
        sequence_output = self.trns(
               encoded["input_ids"],
               attention_mask=encoded["attention_mask"])["last_hidden_state"]
        self.last_hidden_output = sequence_output[:,0,:].view(-1,self.in_dim)

        return self.mlp(sequence_output[:,0,:].view(-1,self.in_dim))

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

def Clinical_Transformer(trns_arch='distilbert', out_dim=1, size=[], batchnorm=True, classification=False):
    return TransformerWithMLP(trns_arch=trns_arch, out_dim=out_dim, size=size, batchnorm=batchnorm)
    
def Genetic_MLP(in_dim, out_dim, size=[1280], batchnorm=True):
    return SimpleMLP(in_dim, out_dim, size=size, batchnorm=batchnorm) 

def TCGAEncoders(data_types=['rna-seq'], dataset=None, mode="validate", rep_dim=None):
    if dataset == None:
        raise ValueError("TCGA Encoder generation method must receive a dataset!")
    if mode == "validate":
        task_is_classification = False

        output_dim = dataset[0]["target"]
        if len(dataset[0]["target"].size()) == 0:
            outputs = dataset[:]["target"]
            if torch.all((outputs - torch.round(outputs)) == 0):
                output_dim = torch.max(outputs) + 1
            task_is_classification = True
        else:
            output_dim = list(dataset[0]["target"].size())[0]
        output_dim = int(output_dim)

        if data_types[0] == "rna-seq":
            input_dim = dataset[0]["left"]
            input_dim = input_dim.size()[0]

            assert(len(data_types) == 1)
            return [Genetic_MLP(input_dim, output_dim)]
        elif data_types[0] == "reports":
            return [Clinical_Transformer(out_dim=output_dim)]
        elif data_types[0] == "toy_reports":
            return [Clinical_Transformer(out_dim=output_dim)]
        else:
            raise NotImplementedError("only rna seq has been implemented for model generation")
    else:
        def encoder_map(dtype, dset):
            if dtype == "rna-seq":
                input_dim = dset.size()[1]
                return Genetic_MLP(input_dim, rep_dim)
            elif dtype == "reports":
                return Clinical_Transformer(out_dim=rep_dim)
            elif dtype == "toy_reports":
                return Clinical_Transformer(out_dim=rep_dim)

        encoders = []

        for dtype, name in zip(data_types, ["rna-seq", "reports"]):
            encoders.append(encoder_map(dtype, dataset[:][name])) 

        return encoders

