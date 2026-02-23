import datetime
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from src.bounding_box import dataset
from src.bounding_box.model import BoardBBox
from src.bounding_box.inference import get_bbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_box_iou(loader, model):
    box_iou = 0.0
    num_samples = 0
    for imgs, target_boxes, target_masks in loader:
        for img, target_box, target_mask in zip(imgs, target_boxes, target_masks):
            output_box = get_bbox(model, img)
            if output_box is not None:
                box_iou += (
                    torchvision.ops.box_iou(
                        output_box.unsqueeze(0), target_box.unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
            num_samples += 1

    return box_iou / num_samples


LOSS_REPORT_FREQ = 50
TEST_ACC_FREQ = 200


def train(
    game: str,
    outdir="models",
    total_steps=10_000,
    batch_size=8,
    max_lr=0.001,
    test_set_size=500,
):
    start_time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(start_time_string)

    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")

    train_set = dataset.GenerativeBboxDataset(
        game=game,
        augment_ratio=0.4,
    )
    test_set = dataset.generate_fixed_test_set(game=game, size=test_set_size)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    model = BoardBBox()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
    )

    test_box_iou_list = []
    best_box_iou = -1.0
    best_model = None
    num_steps = 0

    for img, target_box, target_mask in train_loader:
        img = img.to(device)
        target_mask = target_mask.to(device)

        optimizer.zero_grad()

        output = model(img)
        loss = criterion(output, target_mask)
        loss.backward()
        optimizer.step()
        scheduler.step()

        num_steps += 1

        if num_steps % LOSS_REPORT_FREQ == 0:
            print(
                f"[{num_steps}/{total_steps}] "
                f"loss: {loss.item():.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.5f}"
            )

        if num_steps % TEST_ACC_FREQ == 0 or num_steps >= total_steps:
            test_box_iou = get_box_iou(test_loader, model)
            test_box_iou_list.append(test_box_iou)
            print(f"Num steps: {num_steps}, Test IOU: {test_box_iou_list[-1]:.3f}")

            if test_box_iou > best_box_iou:
                best_box_iou = test_box_iou
                best_model = model.state_dict()
                print(f"Best model updated: Test IOU: {best_box_iou:.3f}")

        if num_steps >= total_steps:
            break

    os.makedirs(outdir, exist_ok=True)
    file_name = f"{outdir}/best_model_bbox_{game}_{best_box_iou:.3f}_{start_time_string}.pth"
    print("Saving to", file_name)
    torch.save(best_model, file_name)

    plt.figure(figsize=(12, 4))
    plt.plot(test_box_iou_list, label="Test IOU")
    plt.xlabel("Step")
    plt.ylabel("IOU")
    plt.legend()
    plt.savefig(file_name + ".png", dpi=250)
