import numbers
from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image
from videsrgan.realesrgan import RealESRGAN

from moviepy.Effect import Effect

@dataclass
class RealESRGANFx(Effect):
    """Effect returning a video clip that is a upscaled version of the clip.

    Parameters
    ----------

    new_size : tuple or float or function, optional
        Can be either
        - ``(width, height)`` in pixels or a float representing
        - A scaling factor, like ``0.5``.
        - A function of time returning one of these.

    height : int, optional
        Height of the new clip in pixels. The width is then computed so
        that the width/height ratio is conserved.

    width : int, optional
        Width of the new clip in pixels. The height is then computed so
        that the width/height ratio is conserved.

    Examples
    --------

    .. code:: python

        clip.with_effects([vfx.Resize((460,720))]) # New resolution: (460,720)
        clip.with_effects([vfx.Resize(0.6)]) # width and height multiplied by 0.6
        clip.with_effects([vfx.Resize(width=800)]) # height computed automatically.
        clip.with_effects([vfx.Resize(lambda t : 1+0.02*t)]) # slow clip swelling
    """

    realesrgan: RealESRGAN = None
    height: int = None
    width: int = None
    apply_to_mask: bool = True

    def upscaler(self, frame):
        """Resize the image using PIL."""
        pil_img = Image.fromarray(frame)
        upscaled_pil = self.realesrgan.process_pil(pil_img)
        return np.array(upscaled_pil)

    def apply(self, clip):
        """Apply the effect to the clip."""
        w, h = clip.size

        if clip.is_mask:

            def image_filter(frame):
                return (1.0 * self.upscaler((255 * frame).astype("uint8")) / 255.0)

        else:

            def image_filter(frame):
                return self.upscaler(frame.astype("uint8"))

        new_clip = clip.image_transform(image_filter)

        if self.apply_to_mask and clip.mask is not None:
            new_clip.mask = clip.mask.with_effects(
                [RealESRGANFx(self.realesrgan, apply_to_mask=False)]
            )

        return new_clip