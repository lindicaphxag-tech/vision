"""Pixel-center registration tests for the real torchvision KeyPoints kernels.

This isolated test extracts the production functions from the source file so
it can run with a CPU PyTorch wheel before the full TorchVision C++ extension
is built. The existing TestResize integration/reference tests are updated too.
"""

import ast
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn.functional as torch_F


SOURCE = Path(__file__).resolve().parents[1] / "torchvision" / "transforms" / "v2" / "functional" / "_geometry.py"


def _real_source_kernels():
    tree = ast.parse(SOURCE.read_text())
    names = {"resize_keypoints", "horizontal_flip_keypoints", "vertical_flip_keypoints"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise AssertionError("Real KeyPoints production functions missing")

    def _size(canvas_size, size, max_size=None):
        if max_size is not None or len(size) != 2:
            raise ValueError("Oracle tests use explicit two-dimensional target sizes")
        return tuple(size)

    ns = {"torch": torch, "Optional": Optional, "_compute_resized_output_size": _size}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), ns)
    return (ns["resize_keypoints"], ns["horizontal_flip_keypoints"], ns["vertical_flip_keypoints"])


resize_keypoints, horizontal_flip_keypoints, vertical_flip_keypoints = _real_source_kernels()


@pytest.mark.parametrize("old_size,new_size", [
    ((65, 97), (94, 155)),
    ((65, 97), (47, 65)),
    ((65, 97), (85, 77)),
    ((7, 13), (15, 21)),
    ((20, 40), (10, 20)),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_pixel_center_mapping_matches_independent_double_precision_reference(old_size, new_size, dtype):
    height, width = old_size
    points = torch.tensor(
        [[0, 0], [width - 1, height - 1], [width / 3, height / 2], [0.25, 0.75]], dtype=dtype
    )
    got, canvas = resize_keypoints(points, size=list(new_size), canvas_size=old_size)
    scale = torch.tensor([new_size[1] / width, new_size[0] / height], dtype=torch.float64)
    # Do not use the implementation's formula to generate its own source data.
    reference = (points.double() + 0.5) * scale - 0.5
    assert canvas == new_size
    assert got.dtype == dtype
    torch.testing.assert_close(got, reference.to(dtype), rtol=0, atol=0)
    assert torch.equal(points, torch.tensor([[0, 0], [width - 1, height - 1], [width / 3, height / 2], [0.25, 0.75]], dtype=dtype))


@pytest.mark.parametrize("old_size,new_size", [
    ((65, 97), (94, 155)),
    ((65, 97), (47, 65)),
    ((7, 13), (15, 21)),
    ((20, 40), (10, 20)),
])
@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_resize_commutes_with_pixel_center_flip(old_size, new_size, axis):
    height, width = old_size
    coords = torch.tensor(
        [[0.0, 0.0], [width - 1.0, height - 1.0], [5.25, 3.75]],
        dtype=torch.float64,
    )
    flip = horizontal_flip_keypoints if axis == "horizontal" else vertical_flip_keypoints
    resize_then_flip = flip(resize_keypoints(coords, list(new_size), old_size)[0], new_size)
    flip_then_resize = resize_keypoints(flip(coords, old_size), list(new_size), old_size)[0]
    torch.testing.assert_close(resize_then_flip, flip_then_resize, rtol=0, atol=1e-13)


@pytest.mark.parametrize("new_size", [(94, 155), (47, 65), (85, 77)])
def test_resized_gaussian_raster_centroid_matches_keypoint_annotation(new_size):
    # Verify against PyTorch's actual align_corners=False pixel interpolation
    # instead of only checking algebraic identities of the keypoint code.
    height, width = 65, 97
    center_x, center_y, sigma = 20.25, 24.5, 3.0
    y = torch.arange(height, dtype=torch.float64)[:, None] + 0.5
    x = torch.arange(width, dtype=torch.float64)[None, :] + 0.5
    image = torch.exp(-((x - center_x).square() + (y - center_y).square()) / (2 * sigma**2))[None, None]
    image_new = torch_F.interpolate(
        image, size=new_size, mode="bilinear", align_corners=False, antialias=True
    )[0, 0]

    yy = torch.arange(new_size[0], dtype=torch.float64) + 0.5
    xx = torch.arange(new_size[1], dtype=torch.float64) + 0.5
    total = image_new.sum()
    measured = torch.stack([
        (image_new.sum(dim=0) * xx).sum() / total,
        (image_new.sum(dim=1) * yy).sum() / total,
    ])
    point = torch.tensor([[center_x - 0.5, center_y - 0.5]], dtype=torch.float64)
    resized = resize_keypoints(point, size=list(new_size), canvas_size=(height, width))[0][0] + 0.5
    torch.testing.assert_close(resized, measured, atol=0.12, rtol=0)


def test_equal_size_preserves_identity_and_empty_points():
    points = torch.tensor([[5.2, 9.8]], dtype=torch.float64)
    result, canvas = resize_keypoints(points, size=[31, 41], canvas_size=(31, 41))
    assert result is points
    assert canvas == (31, 41)
    empty, new_canvas = resize_keypoints(torch.empty((0, 2)), size=[20, 30], canvas_size=(10, 15))
    assert empty.shape == (0, 2)
    assert new_canvas == (20, 30)
