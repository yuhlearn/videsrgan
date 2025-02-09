import pathlib, os
from PIL import Image
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/lib/")
import realesrganbind as bindings

class suppress_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr
    """

    def __init__(self):
        # Open a null file.
        self.null_fd = os.open(os.devnull, os.O_RDWR) 
        # Save the stderr file descriptor.
        self.save_fd = os.dup(2)

    def __enter__(self):
        # Assign the null pointer to stderr.
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        # Re-assign the real stderr back to (2)
        os.dup2(self.save_fd, 2)
        # Close the file descriptors
        os.close(self.null_fd)
        os.close(self.save_fd)

class RealESRGAN:
    def __init__(self, gpuid: int = 0, tta_mode: bool = False, tilesize: int = 0, scale: int = 4, model_path: str = "models", model: str = "RealESRGAN_General_WDN_x4_v3"):
        """
        RealESRGAN class for Super Resolution

        :param gpuid: gpu device to use, -1 for cpu
        :param tta_mode: enable test time argumentation
        :param tilesize: tile size, 0 for auto, must >= 32
        :param model: realesrgan model, 0 for default, -1 for custom load
        """

        # check arguments' validity
        assert gpuid >= -1, "gpuid must >= -1"
        assert tilesize == 0 or tilesize >= 32, "tilesize must >= 32 or be 0"


        self._realesrgan_object = bindings.RealESRGANBind(gpuid, tta_mode)

        self._gpuid = gpuid
        self._tta_mode = tta_mode
        self._tilesize = tilesize
        self._scale = scale
        self._model_path = model_path
        self._model = model

        self.raw_in_image = None
        self.raw_out_image = None

        bindings.create_gpu_instance()

        self._load()

    def _set_parameters(self) -> None:
        """
        Set parameters for RealESRGAN

        :return: None
        """
        self._realesrgan_object.set_parameters(self._tilesize, self._scale)

    def _load(self) -> None:
        """
        Load models from given paths
        :return: None
        """

        model_dir = pathlib.Path(__file__).parent / self._model_path

        param_path = model_dir / pathlib.Path(self._model + ".param")
        model_path = model_dir / pathlib.Path(self._model + ".bin")

        self._set_parameters()

        if param_path is None or model_path is None:
            raise ValueError("param_path and model_path is None")

        self._realesrgan_object.load(str(param_path), str(model_path))

    def process(self) -> None:
        # Suppress all stderr output to avoid printing the default ncnn % progress 
        with suppress_stderr():
            self._realesrgan_object.process(self.raw_in_image, self.raw_out_image)

    def process_pil(self, _image: Image) -> Image:
        """
        Process a PIL image

        :param _image: PIL image
        :return: processed PIL image
        """

        in_bytes = _image.tobytes()
        channels = int(len(in_bytes) / (_image.width * _image.height))
        out_bytes = (self._scale**2) * len(in_bytes) * b"\x00"

        self.raw_in_image = bindings.RealESRGANImage(
            in_bytes, 
            _image.width, 
            _image.height, 
            channels
        )

        self.raw_out_image = bindings.RealESRGANImage(
            out_bytes,
            self._scale * _image.width,
            self._scale * _image.height,
            channels,
        )

        self.process()

        return Image.frombytes(
            _image.mode,
            (
                self._scale * _image.width,
                self._scale * _image.height,
            ),
            self.raw_out_image.get_data(),
        )