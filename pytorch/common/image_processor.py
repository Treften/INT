import cv2
import numpy as np
import torch


class TorchImageProcessor:

    def __init__(
        self,
        image_size,
        is_color,
        mean,
        scale,
        crop_size=0,
        pad=28,
        color="BGR",
        use_cutout=False,
        use_mirroring=False,
        use_random_crop=False,
        use_center_crop=False,
        use_random_gray=False,
    ):
        self.image_size = (
            (image_size, image_size)
            if isinstance(image_size, int)
            else tuple(image_size)
        )
        self.is_color = is_color
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = scale
        self.crop_size = crop_size
        self.pad = pad
        self.color = color.upper()

        self.use_cutout = use_cutout
        self.use_mirroring = use_mirroring
        self.use_random_crop = use_random_crop
        self.use_center_crop = use_center_crop
        self.use_random_gray = use_random_gray

    def _crop(self, image, size, random=False):
        h, w = image.shape[:2]

        if random and self.pad > 0:
            image = cv2.copyMakeBorder(
                image,
                self.pad, self.pad, self.pad, self.pad,
                borderType=cv2.BORDER_REFLECT_101,
            )
            h, w = image.shape[:2]

        if h < size or w < size:
            dh = max(0, size - h)
            dw = max(0, size - w)
            image = cv2.copyMakeBorder(
                image,
                dh // 2, dh - dh // 2,
                dw // 2, dw - dw // 2,
                borderType=cv2.BORDER_REFLECT_101,
            )
            h, w = image.shape[:2]

        if random:
            y = np.random.randint(0, h - size + 1)
            x = np.random.randint(0, w - size + 1)
        else:
            y = (h - size) // 2
            x = (w - size) // 2

        return image[y:y + size, x:x + size]

    def _cutout(self, image):
        h, w = image.shape[:2]
        side = max(1, min(h, w) // 4)

        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)

        y1 = max(0, cy - side // 2)
        y2 = min(h, cy + (side + 1) // 2)
        x1 = max(0, cx - side // 2)
        x2 = min(w, cx + (side + 1) // 2)

        image[y1:y2, x1:x2] = 0
        return image

    def process(self, image_path):
        if isinstance(image_path, (str, bytes)):
            image = cv2.imread(
                image_path,
                cv2.IMREAD_COLOR if self.is_color else cv2.IMREAD_GRAYSCALE,
            )
        else:
            image = np.asarray(image_path).copy()

        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        if self.is_color:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            if self.color == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            if image.ndim == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                elif self.color == "RGB":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.use_mirroring and np.random.random() < 0.5:
            image = cv2.flip(image, 1)

        crop_size = self.crop_size or min(image.shape[:2])

        if self.use_random_crop:
            image = self._crop(image, crop_size, random=True)
        elif self.use_center_crop:
            image = self._crop(image, crop_size, random=False)

        width, height = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        if self.use_random_gray and np.random.random() < 0.2:
            if image.ndim == 3:
                gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY if self.color == "RGB"else cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB if self.color == "RGB"else cv2.COLOR_GRAY2BGR)

        if self.use_cutout and np.random.random() < 0.5:
            image = self._cutout(image)

        image = image.astype(np.float32)
        mean = self.mean

        if mean.ndim == 1 and image.ndim == 3 and len(mean) == image.shape[2]:
            mean = mean.reshape(1, 1, -1)

        image = (image - mean) * self.scale

        if image.ndim == 2:
            image = image[..., np.newaxis]

        return image.astype(np.float32)

