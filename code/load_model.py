import torch
import transformers
import model
import datasets
import json

# script to do very fast (and casual) tests in command line
# use, exec(open("load_model.py").read())

g_dataset = None
g_encoder = None
g_lr = None
g_mode = None
g_tokenier = None

def most_recent_file(folder, ext=""):
    max_time = 0
    max_file = ""
    for dirname, subdirs, files in os.walk(folder):
        for fname in files:
            full_path = os.path.join(dirname, fname)
            time = os.stat(full_path).st_mtime
            if time > max_time and full_path.endswith(ext):
                max_time = time
                max_file = full_path

    return max_file

def load_dataset_and_model(path_dir, num=-1):
    global g_dataset, g_encoder, g_lr, g_mode

    if num == -1:
        print("fetching most recent model (CTRL+C if this isn't what you want!")
        file_cand = most_recent_file("../save/" + path_dir) 
        num = file_cand[file_cand.rfind("/"):file_cand.rfind(".")]
        num = int(num)
        print("num is", num)
        input("confirm")

    with open("../save/" + path_dir + "/" + str(num) + ".args") as f:
        args = json.load(f)
    if args["mode"] == "validate":
        dataset = datasets.TCGADataset(lr_data=[args["left"]], target=args["target"])

        encoder = model.TCGAEncoders(data_types=[args["left"]], dataset=dataset, mode=args["mode"])[0]
        encoder.load_state_dict(torch.load("../save/" + path_dir + "/" + str(num) + ".pth")["state_dict"])

        g_dataset = dataset
        g_encoder = encoder
        g_encoder.eval()
        g_lr = [args["left"]]
        g_mode = args["mode"]

        return
    else:
        raise NotImplementedError("non-validation modes not yet implemented")

def sample_dataset(id):
    global g_dataset, g_encoder, g_lr, g_mode

    if g_mode == "validate":
        idx = g_dataset.tcga_ids.index(id)
        data = g_dataset[idx]["left"]
        print(data)
        for key in data.keys():
            data[key] = torch.unsqueeze(data[key], dim=0)
        out = g_encoder(data)
    else:
        raise NotImplementedError("only validation mode is currently implemented")

    print(out)
    print(g_dataset[idx]["target"])
    return

def try_own(inn):
    global g_dataset, g_encoder, g_lr, g_mode

    if g_mode == "validate":
        print(g_lr)
        if g_lr[0] == "reports":
            dataset = datasets.TCGA_Reports_Dataset(input_txt=inn)
        else:
            raise NotImplementedError("custon rna-seq or other data not yet implemented")
    else:
        raise NotImplementedError("only validation mode is currently implemented")

    out = g_encoder(dataset)
    print(g_encoder.last_hidden_output)

    print(out)
    return
    


