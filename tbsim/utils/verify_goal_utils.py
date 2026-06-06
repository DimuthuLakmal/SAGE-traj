import numpy as np
import torch
from PIL import Image, ImageDraw
from l5kit.geometry import transform_points
from torchvision.utils import save_image


def tensor_to_rgb_image(image_tensor):
    """
    image_tensor: [C, H, W]
    returns: uint8 [H, W, 3]
    """
    img = image_tensor.detach().cpu().float()

    # use first 3 channels if the map has more channels
    if img.shape[0] > 3:
        img = img[:3]

    img = img.permute(1, 2, 0).numpy()

    # normalize only for display
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)

    return (img * 255).astype(np.uint8)


def agent_to_raster_np(points_agent, raster_from_agent):
    """
    Same idea as CTG's agent_to_raster_np.

    points_agent: [N, 2]
    raster_from_agent: [3, 3]
    returns: [N, 2] raster pixel coordinates
    """
    if torch.is_tensor(points_agent):
        points_agent = points_agent.detach().cpu().numpy()

    if torch.is_tensor(raster_from_agent):
        raster_from_agent = raster_from_agent.detach().cpu().numpy()

    return transform_points(points_agent[None], raster_from_agent)[0]


def draw_goal_points_on_map(
    data_batch,
    pred_goals,
    gt_goals=None,
    batch_idx=0,
    sample_idx=0,
    heat_map=None,
    pred_color="#FF6B35",
    gt_color="#2E86DE",
    pred_outline="#8B2500",
    gt_outline="#003C7A",
    point_radius=4,
    draw_index=True
):
    """
    data_batch["image"]: [B, C, 224, 224]
    data_batch["raster_from_agent"]: [B, 3, 3]

    pred_goals:
        [B, K, 3] or [B, N, K, 3]

    gt_goals:
        [B, K, 3], optional

    Returns:
        np.ndarray [H, W, 3] uint8
    """

    image = tensor_to_rgb_image(data_batch["maps"][batch_idx])
    raster_from_agent = data_batch["raster_from_agent"][batch_idx]

    # Select predicted goal sample
    if pred_goals.ndim == 4:
        goals_pred = pred_goals[batch_idx, sample_idx]  # [K, 3]
    elif pred_goals.ndim == 3:
        goals_pred = pred_goals[batch_idx]              # [K, 3]
    else:
        raise ValueError("pred_goals must be [B,K,3] or [B,N,K,3]")

    goals_pred_xy = goals_pred[..., :2]
    pred_pixels = agent_to_raster_np(goals_pred_xy, raster_from_agent)

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    # Draw predicted goals
    for i, p in enumerate(pred_pixels):
        x, y = float(p[0]), float(p[1])

        circle = [
            x - point_radius,
            y - point_radius,
            x + point_radius,
            y + point_radius,
        ]

        draw.ellipse(circle, fill=pred_color, outline=pred_outline)

        if draw_index:
            draw.text((x + point_radius + 2, y + point_radius + 2), f"P{i}", fill=pred_outline)

    # Draw GT goals if provided
    if gt_goals is not None:
        goals_gt = gt_goals[batch_idx]  # [K, 3]
        goals_gt_xy = goals_gt[..., :2]
        gt_pixels = agent_to_raster_np(goals_gt_xy, raster_from_agent)

        for i, p in enumerate(gt_pixels):
            x, y = float(p[0]), float(p[1])

            # draw cross
            r = point_radius + 2
            draw.line([x - r, y - r, x + r, y + r], fill=gt_color, width=2)
            draw.line([x - r, y + r, x + r, y - r], fill=gt_color, width=2)

            if draw_index:
                draw.text((x + r + 2, y - r - 2), f"G{i}", fill=gt_color)

    # save_image(heat_map[0], 'heat_map.png')
    Image.fromarray(np.asarray(im)).save("goal_overlay.png")