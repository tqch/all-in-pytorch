import os
import torch
import torch.nn.functional as F
from utils import History, save_fig, save_audio
from tqdm import tqdm
from collections import OrderedDict


def train(
        model,
        trainloader,
        testloader,
        epochs,
        optimizer,
        loss_fn,
        device,
        data_type="image",
        preprocessing=None,
        postprocessing=None,
        additional_eval=None,
        num_batches_eval=-1,
        index_type=None,
        save_dir=".",
        **kwargs
):
    history = History(index_type)
    for e in range(epochs):
        with tqdm(trainloader, desc=f"{e + 1}/{epochs} epochs") as t:
            running_loss = 0
            running_additionals = [0 for _ in range(len(history.get_indices())-1)]
            running_total = 0
            for i, (x, y) in enumerate(t):
                model.train()
                if preprocessing is not None:
                    x_preprocessed = preprocessing(x.to(device))
                else:
                    x_preprocessed = x
                if hasattr(model, "use_label"):
                    x_hat = model(x_preprocessed.to(device), y.to(device))
                else:
                    x_hat = model(x_preprocessed.to(device))
                loss = loss_fn(x_hat, x.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
                running_total += x.size(0)

                if additional_eval is not None:
                    additionals = additional_eval(
                        torch.sigmoid(x_hat.detach().cpu()).numpy(),
                        x.numpy()
                    )
                    for k in range(len(running_additionals)):
                        running_additionals[k] += additionals[k] * x.size(0)

                if i < len(trainloader) - 1:
                    t.set_postfix(OrderedDict([
                        (index, value)
                        for index, value in zip(
                            list(map(lambda x: "train_"+x, history.get_indices())),
                            list(map(lambda x: x/running_total, [running_loss, ] + running_additionals))
                        )
                    ]))
                else:
                    test_loss = 0
                    test_total = 0
                    test_additionals = [0 for _ in range(len(history.get_indices())-1)]
                    for j, (x, y) in enumerate(testloader):
                        model.eval()
                        if preprocessing is not None:
                            x_preprocessed = preprocessing(x)
                        else:
                            x_preprocessed = x
                        with torch.no_grad():
                            if hasattr(model, "use_label"):
                                x_hat = model(x_preprocessed.to(device), y.to(device))
                            else:
                                x_hat = model(x_preprocessed.to(device))
                            test_loss += loss_fn(x_hat, x.to(device)).item() * x.size(0)
                            test_total += x.size(0)
                            if additional_eval is not None:
                                additionals = additional_eval(
                                    torch.sigmoid(x_hat.cpu()).numpy(),
                                    x.numpy()
                                )
                                for k in range(len(test_additionals)):
                                    test_additionals[k] += additionals[k] * x.size(0)
                        if j == 0:
                            if data_type == "audio":
                                x_hat = model.generate(
                                    x[:kwargs["n_tracks"]].to(device),
                                    F.one_hot(
                                        kwargs["person_id"] * torch.ones(kwargs["n_tracks"]).long().to(device),
                                        model.num_voices
                                    ))
                                if postprocessing is not None:
                                    audio = postprocessing(x_hat).detach().cpu()
                                else:
                                    audio = x_hat.detach().cpu()
                                save_audio(
                                    audio,
                                    e + 1,
                                    sampling_rate=kwargs["sampling_rate"],
                                    save_folder=os.path.join(
                                        save_dir,
                                        "audios",
                                        model.__module__.split(".")[-1]
                                    )
                                )
                            elif data_type == "image":
                                if hasattr(model, "generate"):
                                    if hasattr(kwargs, "occlude"):
                                        img = model.generate(x[:kwargs["n_plots"], :, :kwargs["occlude"], :].to(device))
                                    else:
                                        img = model.generate(n_samples=kwargs["n_plots"])
                                else:
                                    img = torch.sigmoid(x_hat[:kwargs["n_plots"]].detach().cpu())
                                save_fig(
                                    img,
                                    e + 1,
                                    save_folder=os.path.join(
                                        save_dir,
                                        "figures",
                                        model.__module__.split(".")[-1]
                                    ))
                        elif num_batches_eval > 0 and j == num_batches_eval - 1:
                            break
                    train_epoch_dict = history.index_type(*list(
                        map(
                            lambda x: x / running_total,
                            [running_loss, ] + running_additionals
                        )))
                    test_epoch_dict = history.index_type(*list(
                        map(
                            lambda x: x / test_total,
                            [test_loss, ] + test_additionals
                        )))
                    history.update_history(train_epoch_dict, test_epoch_dict)
                    t.set_postfix(history.get_last_epoch())
