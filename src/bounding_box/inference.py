import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bbox(model, img: torch.Tensor):
    model.eval()
    model.to(device)
    with torch.no_grad():
        assert (
            len(img.shape) == 3
        ), "Need input to be of shape [C, H, W] but is: " + str(img.shape)
        assert img.shape[0] == 3, "Channel dimension must be 3 (RGB)"
        img = img.unsqueeze(0)
        mask = (model(img.to(device)) > 0.5).float().cpu().squeeze(0)

        if mask.sum() == 0:
            return None
        output_box = torchvision.ops.masks_to_boxes(mask)

    model.train()
    return output_box.squeeze(0)
