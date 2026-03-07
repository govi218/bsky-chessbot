import torch

from src import consts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mask_to_corners(mask: torch.Tensor) -> torch.Tensor | None:
    """Extract 4 quad corners from a binary mask [H, W].

    Returns [4, 2] tensor of (x, y) pixel coordinates in order TL, TR, BR, BL,
    or None if the mask is empty.
    """
    ys, xs = torch.where(mask > 0.5)
    if len(xs) == 0:
        return None

    xs_f = xs.float()
    ys_f = ys.float()

    # Find extremal points in each diagonal direction
    tl_idx = (xs_f + ys_f).argmin()  # minimize x+y
    tr_idx = (xs_f - ys_f).argmax()  # maximize x-y
    br_idx = (xs_f + ys_f).argmax()  # maximize x+y
    bl_idx = (ys_f - xs_f).argmax()  # maximize y-x

    corners = torch.tensor([
        [xs[tl_idx].item(), ys[tl_idx].item()],
        [xs[tr_idx].item(), ys[tr_idx].item()],
        [xs[br_idx].item(), ys[br_idx].item()],
        [xs[bl_idx].item(), ys[bl_idx].item()],
    ], dtype=torch.float32)

    return corners


def get_quad(model, img: torch.Tensor):
    """Run inference and return 4 corner points, or None.

    Accepts an image tensor in [0, 1] range with shape [C, H, W].
    Returns corners as [4, 2] tensor in BBOX_IMAGE_SIZE pixel coordinates.
    Corner order: TL, TR, BR, BL.
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        assert (
            len(img.shape) == 3
        ), "Need input to be of shape [C, H, W] but is: " + str(img.shape)
        assert img.shape[0] == 3, "Channel dimension must be 3 (RGB)"

        logits = model(img.unsqueeze(0).to(device))  # [1, 1, H, W]
        mask = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu()  # [H, W]

        corners = mask_to_corners(mask)

    return corners, mask
