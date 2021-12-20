import torch

import datasets
import model

tasks = ["papillary", "lv", "perineural", "metastasis", "papillary_nl", "lv_nl", "perineural_nl", "metastasis_nl"]
contrastive = ["rna-seq", "clinical"]
datahandler = datasets.TCGADataHandler(contrastive=["rna-seq", "clinical"], zero_shot=[], finetune=tasks, train_ratio=0, ft_train_ratio=0.9, lg_types=True) 
encoders = model.TCGAEncoders(data_types=["rna-seq", "clinical"], datahandler=datahandler, mode=["evaluate"], rep_dim=32)
dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)

print(f"[Evaluate] begin evaluation on tasks {tasks}")

for it, task in enumerate(tasks):
    #if finetune:
    print(f"[Evaluate] begin task {task} evaluation ({it} out of {len(tasks)})")

    train_loader = torch.utils.data.DataLoader(
            dataset=datahandler.val_train[task],
            shuffle=True,
            batch_size=32,
            **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(
            dataset=datahandler.val_test[task],
            shuffle=True,
            batch_size=32,
            **dataloader_kwargs)

    if "_nl" not in task:
        output_nclasses = int(max(torch.max(datahandler.val_train[task][:][task]).item(), torch.max(datahandler.val_test[task][:][task]).item()) + 1)

        ft_encoder = model.WithFinetuneLayers(encoders[encoder_idx], 32, output_size=2, size=[])
        ft_encoders.append(ft_encoder)

        if finetune:
            model_optimizer = torch.optim.AdamW(ft_encoder.model.parameters(), lr=0.001, weight_decay=0.0001)
        ft_optimizer = torch.optim.AdamW(ft_encoder.finetune.parameters(), lr=0.001, weight_decay=0.0001)

        for e in range(1, 20):
            # evaluate
            ft_encoder.eval()

            correct = 0
            total = 0
            saved_loss = []
            with torch.no_grad():
                for it, elem in enumerate(test_loader):
                    out = ft_encoder(elem[contrastive[encoder_idx]])

                    if output_nclasses != 2:
                        loss = torch.nn.CrossEntropyLoss()
                        l = loss(out, elem[task].long())
                    else:
                        loss = torch.nn.BCEWithLogitsLoss()
                        l = loss(out, torch.nn.functional.one_hot(elem[task].long(), num_classes=output_nclasses).float())
                    saved_loss.append(float(l.item()))

                    out = torch.argmax(out, dim=1)
                    correct += torch.numel(torch.where(out == elem[task])[0])
                    total += int(elem[list(elem.keys())[0]].size()[0])
    else:
        ft_encoder = deepcopy(encoders[encoder_idx])
        if "reports" in contrastive:
            nl_encoder = encoders[contrastive.index("reports")]
        elif "clean-reports" in args.contrastive:
            nl_encoder = encoders[contrastive.index("clean-reports")]
        else:
            raise ValueError("nl target but can't find nl encoder")
        
        ft_optimizer = torch.optim.AdamW(ft_encoder.parameters(), lr=0.001, weight_decay=0.0001)
        nl_optimizer = torch.optim.AdamW(nl_encoder.parameters(), lr=0.001, weight_decay=args.wd)

        for e in range(1, 2 + args.ft_epochs):
            # evaluate
            ft_encoder.eval()
            nl_encoder.eval()

            acc = []
            with torch.no_grad():
                for it, elem in enumerate(test_loader):
                    ft_out = ft_encoder(elem[args.contrastive[encoder_idx]])
                    nl_out = nl_encoder(elem[task])

                    acc.append(clip_acc(ft_out, nl_out, distance=distance, remove_duplicates=True))
            acc = np.mean(np.array(acc))
            print(f"[Evaluate] on epoch {e - 1}, task {task}: accuracy {round(float(acc), 3)}")
            file_to_update.write(f"epoch {e - 1} & task {task} & zero shot: accuracy {round(float(acc), 3)}")

            if e > args.ft_epochs:
                if save_num != -1:
                    torch.save(dict(epoch=0, state_dict=ft_encoder.state_dict()), os.path.join(args.path_dir, str(save_num) + "-" + task + ".pth"))
                break

            # train
            ft_encoder.train()
            nl_encoder.train()

            print(f"[Evaluate] nl-epoch: {e} out of {args.ft_epochs}")
            for it, elem in enumerate(train_loader):
                ft_out = ft_encoder(elem[args.contrastive[encoder_idx]])
                nl_out = nl_encoder(elem[task])

                l = _uni_info_nce(ft_out, nl_out, distance=distance, remove_duplicates=True, both_sides=False)

                l.backward()
                if finetune:
                    model_optimizer.step()
                ft_optimizer.step()
    # else:
    #     output_nclasses = int(max(torch.max(datahandler.val_train[task][:][task]).item(), torch.max(datahandler.val_test[task][:][task]).item()) + 1)
