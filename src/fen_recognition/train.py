import datetime
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from src import common, consts
from src.fen_recognition import dataset
from src.fen_recognition.model import BoardRec
from src.games import get_game

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accuracy_and_loss(loader, model, criterion, game: str):
    spec = get_game(game)
    num_correct = 0
    num_samples = 0
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for img, target in loader:
            img = img.to(device)
            target = target.to(device)
            output = model(img)

            loss += criterion(output, target).item() * target.size(0)

            output = output.cpu()
            target = target.cpu()

            assert output.size(0) == target.size(0)
            for i in range(0, output.size(0)):
                if common.tensor_to_position(output[i], spec).piece_placement == common.tensor_to_position(target[i], spec).piece_placement:
                    num_correct += 1

            num_samples += target.size(0)
    model.train()
    return num_correct / num_samples, loss / num_samples


LOSS_REPORT_FREQ = 200
TEST_ACC_FREQ = 4000


def train(
    game: str,
    data_root_dir=None,
    outdir="models",
    total_steps=600_000,
    batch_size=8,
    max_lr=0.001,
    train_test_split=0.97,
    lr_schedule_pct_start=0.3,
    max_data=None,
    checkpoint=None,
    tile_size: int = consts.DEFAULT_TILE_SIZE,
):
    spec = get_game(game)
    if data_root_dir is None:
        data_root_dir = f"resources/board_position_images/{spec.key}"

    start_time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(start_time_string)
    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")
    print("Game:", spec.key)

    board_set = dataset.BoardPositionDataset(
        root_dir=data_root_dir,
        game=spec,
        tile_size=tile_size,
        augment_ratio=0.8,
        affine_augment_ratio=0.8,
        max=max_data,
        device=device,
    )
    train_set, test_set = torch.utils.data.random_split(board_set, [train_test_split, 1.0 - train_test_split])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    model = BoardRec(game=spec.key, tile_size=tile_size)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
        print("Using checkpoint:", checkpoint)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=lr_schedule_pct_start,
    )

    test_loss_list = []
    test_acc_list = []
    best_acc = -1.0
    best_model = None
    num_steps = 0

    while num_steps < total_steps:
        running_loss = 0.0

        for i, (img, target) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(img)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            num_steps += 1
            running_loss += loss.item()

            if (i + 1) % LOSS_REPORT_FREQ == 0:
                print(
                    f"[{num_steps}/{total_steps}, {i + 1:5d}] "
                    f"loss: {running_loss / LOSS_REPORT_FREQ:.4f}, "
                    f"lr: {optimizer.param_groups[0]['lr']:.5f}"
                )
                running_loss = 0.0

            if (i + 1) % TEST_ACC_FREQ == 0 or num_steps >= total_steps:
                test_acc, test_loss = get_accuracy_and_loss(test_loader, model, criterion, game=spec.key)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                print(
                    f"Num steps: {num_steps}, "
                    f"Test Loss: {test_loss_list[-1]:.4f}, "
                    f"Test Acc: {test_acc_list[-1]:.3f}"
                )

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model = model.state_dict()
                    print(f"Best model updated: Test Acc: {best_acc:.3f}")

            if num_steps >= total_steps:
                break

    os.makedirs(outdir, exist_ok=True)
    file_name = f"{outdir}/best_model_position_{spec.key}_{best_acc:.3f}_{start_time_string}.pth"
    print("Saving to", file_name)
    torch.save(best_model, file_name)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(test_loss_list, label="Test Loss")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_acc_list, label="Test Acc")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(file_name + ".png", dpi=250)
