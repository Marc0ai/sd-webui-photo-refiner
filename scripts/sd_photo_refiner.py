import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageChops, ImageFilter

class Script(scripts.Script):  
    def title(self):
        return "Photo Refiner"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        blur_intensity = gr.Slider(minimum=0, maximum=5, step=0.1, value=0, label="Blur")
        sharpen_intensity = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Sharpening")
        chromatic_aberration = gr.Slider(minimum=0, maximum=3, step=1, value=0, label="Chromatic Aberration")
        saturation_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Saturation")
        contrast_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Contrast")
        brightness_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Brightness")
        highlights_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Highlights")
        shadows_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Shadows")
        auto_rgb = gr.Checkbox(value=False, label="Automatic Tones")
        film_grain = gr.Checkbox(value=False, label="Filmic Grain")
        return [auto_rgb, blur_intensity, sharpen_intensity, chromatic_aberration, saturation_intensity, contrast_intensity, brightness_intensity, highlights_intensity, shadows_intensity, film_grain]

    def run(self, p, auto_rgb, blur_intensity, sharpen_intensity, chromatic_aberration, saturation_intensity, contrast_intensity, brightness_intensity, highlights_intensity, shadows_intensity, film_grain):
        def apply_effects(im, auto_rgb, blur, sharpen, ca, saturation, contrast, brightness, highlights, shadows, film_grain):
            if isinstance(im, np.ndarray):
                img = Image.fromarray(im)
            else:
                img = im
                
            if auto_rgb:
                im = np.array(img)
                mean_r = np.mean(im[:, :, 0])
                mean_g = np.mean(im[:, :, 1])
                mean_b = np.mean(im[:, :, 2])

                target_r, target_g, target_b = 118, 118, 113

                r_factor = target_r / mean_r
                g_factor = target_g / mean_g
                b_factor = target_b / mean_b

                im[:, :, 0] = np.clip(im[:, :, 0] * r_factor, 0, 255)
                im[:, :, 1] = np.clip(im[:, :, 1] * g_factor, 0, 255)
                im[:, :, 2] = np.clip(im[:, :, 2] * b_factor, 0, 255)

                img = Image.fromarray(im)                

            if blur > 0:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur))

            if sharpen > 0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(sharpen)

            if ca > 0:
                r, g, b = img.split()
                r = ImageChops.offset(r, ca, 0)
                b = ImageChops.offset(b, -ca, 0)
                img = Image.merge('RGB', (r, g, b))

            if saturation != 0:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(saturation / 5.0 + 1)

            if contrast != 0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast / 5.0 + 1)

            if brightness != 0:
                img = Image.fromarray(np.clip(np.array(img) * (brightness / 5.0 + 1), 0, 255).astype(np.uint8))

            if highlights != 0 or shadows != 0:
                img_np = np.array(img).astype(np.float32) / 255.0

                if shadows != 0:
                    gamma = 1.0 / (shadows / 5.0 + 1) if shadows > 0 else 1.0 + abs(shadows / 5.0)
                    img_np = np.power(img_np, gamma)

                if highlights != 0:
                    highlights_scale = highlights / 5.0 + 1
                    img_np[img_np > 0.5] = 0.5 + (img_np[img_np > 0.5] - 0.5) * highlights_scale

                img = Image.fromarray(np.clip(img_np * 255, 0, 255).astype(np.uint8))

            if film_grain:
                grain = np.random.normal(0, 1, (img.height, img.width))
                grain = np.clip(grain * 255, 0, 255).astype(np.uint8)
                grain_img = Image.fromarray(grain).convert('L')
                grain_img = grain_img.resize((img.width, img.height), Image.NEAREST)
                grain_img = grain_img.filter(ImageFilter.GaussianBlur(radius=1))
                img = Image.blend(img.convert('RGB'), grain_img.convert('RGB'), alpha=0.04)

            return img

        proc = process_images(p)

        for i in range(len(proc.images)):
            proc.images[i] = apply_effects(proc.images[i], auto_rgb, blur_intensity, sharpen_intensity, chromatic_aberration, saturation_intensity, contrast_intensity, brightness_intensity, highlights_intensity, shadows_intensity, film_grain)
            images.save_image(proc.images[i], p.outpath_samples, "",
                              proc.seed + i, proc.prompt, opts.samples_format, info=proc.info, p=p)

        return proc
