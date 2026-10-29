
import cv2
import numpy as np


class TorchImageProcessor:
    def __init__(
        self, image_size, is_color, mean, scale, crop_size=0, pad=28,
        color="BGR", use_cutout=False, use_mirroring=False,
        use_random_crop=False, use_center_crop=False, use_random_gray=False,
    ):
        self.image_size = tuple(image_size)
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

    def _crop(self, image, size, random_crop=False):
        h, w = image.shape[:2]

        if random_crop and self.pad:
            image = cv2.copyMakeBorder(
                image, self.pad, self.pad, self.pad, self.pad,
                cv2.BORDER_REFLECT_101,
            )
            h, w = image.shape[:2]

        if h < size or w < size:
            dh, dw = max(0, size - h), max(0, size - w)
            image = cv2.copyMakeBorder(
                image, dh // 2, dh - dh // 2,
                dw // 2, dw - dw // 2, cv2.BORDER_REFLECT_101,
            )
            h, w = image.shape[:2]

        if random_crop:
            y = np.random.randint(h - size + 1)
            x = np.random.randint(w - size + 1)
        else:
            y, x = (h - size) // 2, (w - size) // 2

        return image[y:y + size, x:x + size]

    def process(self, image_path):
        if isinstance(image_path, (str, bytes)):
            flag = cv2.IMREAD_COLOR if self.is_color else cv2.IMREAD_GRAYSCALE
            image = cv2.imread(image_path, flag)
        else:
            image = np.asarray(image_path).copy()

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        if self.is_color:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if self.color == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3:
            code = cv2.COLOR_RGB2GRAY if self.color == "RGB" else cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, code)

        if self.use_mirroring and np.random.rand() < 0.5:
            image = cv2.flip(image, 1)

        crop = self.crop_size
        if crop > 0 and self.use_random_crop:
            image = self._crop(image, crop, random_crop=True)
        elif crop > 0 and self.use_center_crop:
            image = self._crop(image, crop)

        width, height = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        if self.is_color and self.use_random_gray and np.random.rand() < 0.2:
            code = cv2.COLOR_RGB2GRAY if self.color == "RGB" else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(image, code)
            code = cv2.COLOR_GRAY2RGB if self.color == "RGB" else cv2.COLOR_GRAY2BGR
            image = cv2.cvtColor(gray, code)

        if self.use_cutout and np.random.rand() < 0.5:
            h, w = image.shape[:2]
            side = max(1, min(h, w) // 4)
            y = np.random.randint(h)
            x = np.random.randint(w)
            image[max(0, y-side//2):min(h, y+side//2),
                  max(0, x-side//2):min(w, x+side//2)] = 0

        image = image.astype(np.float32)
        if self.mean.ndim == 1 and image.ndim == 3:
            mean = self.mean.reshape(1, 1, -1)
        else:
            mean = self.mean
        image = (image - mean) * self.scale

        if image.ndim == 2:
            image = image[..., None]

        return image.astype(np.float32)
