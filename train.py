import os
import torch
from utils import *
from tqdm import tqdm


def train(
        model,
        trainloader,
        testloader,
        epochs,
        optimizer,
        loss_fn,
        device,
        n_plots,
        preprocessing=None,
        save_dir="."
):
    history = History(ImageQuality)
    for e in range(epochs):
        with tqdm(trainloader, desc=f"{e + 1}/{epochs} epochs") as t:
            running_loss = 0
            running_mse = 0
            running_psnr = 0
            running_ssim = 0
            running_total = 0
            for i, (x, _) in enumerate(t):
                model.train()
                if preprocessing is not None:
                    x_preprocessed = preprocessing(x)
                else:
                    x_preprocessed = x
                x_hat = model(x_preprocessed.to(device))
                loss = loss_fn(x_hat, x.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_total += x.size(0)
                mse, psnr, ssim = assess_img_quality(
                    torch.sigmoid(x_hat.detach().cpu()).numpy(),
                    x.numpy()
                )
                running_mse += mse * x.size(0)
                running_psnr += psnr * x.size(0)
                running_ssim += ssim * x.size(0)
                if i < len(trainloader) - 1:
                    t.set_postfix({
                        "train_reconstruction_loss": running_loss / running_total,
                        "train_mse": running_mse / running_total,
                        "train_psnr": running_psnr / running_total,
                        "train_ssim": running_ssim / running_total
                    })
                else:
                    test_loss = 0
                    test_total = 0
                    test_mse = 0
                    test_psnr = 0
                    test_ssim = 0
                    for j, (x, _) in enumerate(testloader):
                        model.eval()
                        x, _ = next(iter(testloader))
                        if preprocessing is not None:
                            x_preprocessed = preprocessing(x)
                        else:
                            x_preprocessed = x
                        with torch.no_grad():
                            x_hat = model(x_preprocessed.to(device))
                            test_loss += loss_fn(x_hat, x.to(device)).item()
                            test_total += x.size(0)
                            mse, psnr, ssim = assess_img_quality(
                                torch.sigmoid(x_hat.detach().cpu()).numpy(),
                                x.numpy()
                            )
                            test_mse += mse * x.size(0)
                            test_psnr += psnr * x.size(0)
                            test_ssim += ssim * x.size(0)
                        if j == 0:
                            save_fig(
                                torch.sigmoid(x_hat[:n_plots].detach().cpu()),
                                e + 1,
                                save_folder=os.path.join(
                                    save_dir,
                                    "figures",
                                    model.__module__.split(".")[-1]
                                ))
                    train_epoch_dict = ImageQuality(*list(
                        map(
                            lambda x: x / running_total,
                            [running_loss, running_mse, running_psnr, running_ssim]
                        )))
                    test_epoch_dict = ImageQuality(*list(
                        map(
                            lambda x: x / test_total,
                            [test_loss, test_mse, test_psnr, test_ssim]
                        )))
                    history.update_history(train_epoch_dict, test_epoch_dict)
                    t.set_postfix(history.get_last_epoch())
