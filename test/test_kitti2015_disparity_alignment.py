"""KITTI2015 input/ground-truth pairing regressions (no large archive download)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from torchvision.datasets import Kitti2015Stereo


def write_image(directory: Path, name: str, *, disparity: bool = False, value: int = 1):
    directory.mkdir(parents=True, exist_ok=True)
    if disparity:
        array = np.full((3, 4), value * 256, dtype=np.uint16)
    else:
        array = np.full((3, 4, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(directory / name)


def fill_stereo(root: Path, split: str, names: tuple[str, ...]):
    for name in names:
        for camera in ("image_2", "image_3"):
            write_image(root / "Kitti2015" / split / camera, name, value=5)


def fill_disparity(root: Path, split: str, names: tuple[str, ...]):
    for name in names:
        for camera in ("disp_occ_0", "disp_occ_1"):
            write_image(root / "Kitti2015" / split / camera, name, disparity=True, value=7)


def test_train_ignores_unlabeled_next_frame_and_keeps_exact_disparity_id(tmp_path):
    # The actual KITTI2015 structure contains both _10 and _11 frame images,
    # while ground-truth disparity labels only exist for the _10 frames.
    fill_stereo(tmp_path, "training",
        ("000000_10.png", "000000_11.png", "000001_10.png", "000001_11.png"))
    fill_disparity(tmp_path, "training", ("000000_10.png", "000001_10.png"))

    dataset = Kitti2015Stereo(tmp_path, split="train")
    assert len(dataset) == 2  # NOT four input frames.
    assert [Path(pair[0]).name for pair in dataset._images] == ["000000_10.png", "000001_10.png"]
    assert [Path(pair[0]).name for pair in dataset._disparities] == [
        Path(pair[0]).name for pair in dataset._images
    ]
    for index in range(len(dataset)):
        image_left, image_right, disparity, mask = dataset[index]
        assert image_left.size == image_right.size == (4, 3)
        np.testing.assert_array_equal(disparity, np.full((1, 3, 4), 7.0))
        assert mask is None


def test_kitti2015_testing_uses_only_benchmark_ten_frames(tmp_path):
    fill_stereo(tmp_path, "testing", ("000000_10.png", "000000_11.png"))
    dataset = Kitti2015Stereo(tmp_path, split="test")
    assert len(dataset) == 1
    image_left, image_right, disparity, mask = dataset[0]
    assert disparity is None and mask is None
    assert image_left.size == image_right.size == (4, 3)


def test_right_camera_scene_shift_fails_instead_of_silent_pairing(tmp_path):
    root = tmp_path / "Kitti2015" / "training"
    for name in ("000000_10.png", "000001_10.png"):
        write_image(root / "image_2", name)
    for name in ("000000_10.png", "000002_10.png"):
        write_image(root / "image_3", name)
    fill_disparity(tmp_path, "training", ("000000_10.png", "000001_10.png"))
    with pytest.raises(ValueError, match="Different left/right KITTI2015 image IDs"):
        Kitti2015Stereo(tmp_path)


def test_right_disparity_scene_shift_fails_closed(tmp_path):
    root = tmp_path / "Kitti2015" / "training"
    fill_stereo(tmp_path, "training", ("000000_10.png", "000001_10.png"))
    write_image(root / "disp_occ_0", "000000_10.png", disparity=True)
    write_image(root / "disp_occ_1", "000001_10.png", disparity=True)
    with pytest.raises(ValueError, match="Different left/right KITTI2015 disparity IDs"):
        Kitti2015Stereo(tmp_path)


def test_disparity_without_corresponding_input_fails_instead_of_wrong_label(tmp_path):
    fill_stereo(tmp_path, "training", ("000000_10.png",))
    fill_disparity(tmp_path, "training", ("000002_10.png",))
    with pytest.raises(ValueError, match="has no matching stereo image pair"):
        Kitti2015Stereo(tmp_path)
