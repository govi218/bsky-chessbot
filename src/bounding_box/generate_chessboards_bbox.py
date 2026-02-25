import random
from pathlib import Path

from PIL import Image

from src import consts, common
from src.fen_recognition.generate_chessboards import BoardGenerator

MIN_CROP_SIZE = consts.BOARD_PIXEL_WIDTH + 10
MAX_CROP_SIZE = consts.BOARD_PIXEL_WIDTH * 4

TARGET_SIZE = consts.BOARD_PIXEL_WIDTH * 2


class BboxGenerator:
    def __init__(
        self,
        game: str,
        background_root_dir: str = "resources/website_screenshots",
        board_middleground_probability: float = 0.4,
    ):
        self.board_generator = BoardGenerator(game)
        self.board_middleground_probability = board_middleground_probability

        background_root_dir = Path(background_root_dir)
        assert background_root_dir.is_dir(), f"With background_root_dir = {background_root_dir}"
        self.background_image_files = common.glob_all_image_files_recursively(background_root_dir)
        assert len(self.background_image_files) > 0, "No background images found"

    def generate_one(self) -> tuple[Image.Image, tuple[int, int, int, int]]:
        # Generate a board image in-memory
        board_image, _ = self.board_generator.generate_one()

        # Pick random background screenshot and crop
        bg_path = random.choice(self.background_image_files)
        img = Image.open(bg_path).convert("RGB")

        img_width, img_height = img.size
        max_size = min(img_width, img_height, MAX_CROP_SIZE)
        crop_width = random.randint(MIN_CROP_SIZE, max_size)
        crop_height = random.randint(MIN_CROP_SIZE, max_size)

        x = random.randint(0, img_width - crop_width)
        y = random.randint(0, img_height - crop_height)
        img = img.crop((x, y, x + crop_width, y + crop_height))

        # Resize board to fit within crop
        board_image_width, board_image_height = board_image.size
        board_aspect = board_image_width / max(board_image_height, 1)
        max_w = min(crop_width, max(4, int(crop_height * board_aspect)))
        max_h = min(crop_height, max(4, int(crop_width / max(board_aspect, 1e-6))))
        new_width = random.randint(max(4, min(max_w, board_image_width // 2)), max_w)
        new_height = max(4, int(new_width / max(board_aspect, 1e-6)))
        if new_height > max_h:
            new_height = max_h
            new_width = max(4, int(new_height * board_aspect))
        board_image = board_image.resize((new_width, new_height))
        board_image_width, board_image_height = board_image.size

        assert board_image_width <= crop_width
        assert board_image_height <= crop_height

        board_x = random.randint(0, crop_width - board_image_width)
        board_y = random.randint(0, crop_height - board_image_height)

        center_x = board_x + board_image_width // 2
        center_y = board_y + board_image_height // 2

        if random.uniform(0.0, 1.0) < self.board_middleground_probability:
            max_relative_size = random.uniform(0.1, 3.0)
            max_aspect_ratio = random.uniform(1.0, 4.0)
            while True:
                middleground_width = random.randint(
                    min(crop_width, 4, board_image_width // 2), crop_width
                )
                middleground_height = random.randint(
                    min(crop_height, 4, board_image_height // 2), crop_height
                )
                middleground_x = random.randint(0, crop_width - middleground_width)
                middleground_y = random.randint(0, crop_height - middleground_height)

                if (
                    middleground_x > board_x
                    and middleground_y > board_y
                    and middleground_x + middleground_width < board_x + board_image_width
                    and middleground_y + middleground_height < board_y + board_image_height
                ):
                    continue

                if (
                    board_image_width * board_image_height * max_relative_size
                    < middleground_height * middleground_width
                ):
                    continue

                if (
                    max(
                        middleground_height / middleground_width,
                        middleground_width / middleground_height,
                    )
                    > max_aspect_ratio
                ):
                    continue

                if (
                    middleground_x > board_x + board_image_width
                    or board_x > middleground_x + middleground_width
                ):
                    continue
                if (
                    middleground_y > board_y + board_image_height
                    or board_y > middleground_y + middleground_height
                ):
                    continue

                # Generate middleground board in-memory
                middleground_img, _ = self.board_generator.generate_one()
                middleground_img = middleground_img.resize(
                    (middleground_width, middleground_height)
                )
                img.paste(middleground_img, (middleground_x, middleground_y))
                break

        img.paste(board_image, (board_x, board_y))

        scale_x = TARGET_SIZE / crop_width
        scale_y = TARGET_SIZE / crop_height

        img = img.resize((TARGET_SIZE, TARGET_SIZE))
        center_x = int(center_x * scale_x)
        center_y = int(center_y * scale_y)
        board_image_width = int(board_image_width * scale_x)
        board_image_height = int(board_image_height * scale_y)

        # Randomly rotate both image and bbox target by right angles.
        rotation = random.choice((0, 90, 180, 270))
        if rotation == 90:
            img = img.transpose(Image.ROTATE_90)
            center_x, center_y = center_y, TARGET_SIZE - 1 - center_x
            board_image_width, board_image_height = board_image_height, board_image_width
        elif rotation == 180:
            img = img.transpose(Image.ROTATE_180)
            center_x, center_y = TARGET_SIZE - 1 - center_x, TARGET_SIZE - 1 - center_y
        elif rotation == 270:
            img = img.transpose(Image.ROTATE_270)
            center_x, center_y = TARGET_SIZE - 1 - center_y, center_x
            board_image_width, board_image_height = board_image_height, board_image_width

        return img, (center_x, center_y, board_image_width, board_image_height)
