import gradio as gr
import os
import modules.scripts as scripts
from modules import images
from modules.shared import opts
from modules.processing import Processed
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageChops, ImageFilter, ImageDraw


class Script(scripts.Script):

    def title(self, enabled=False):
        if enabled:
            return "Photo Refiner - Enabled"
        else:
            return "Photo Refiner"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Blocks() as demo:
            with gr.Accordion(label=self.title(), elem_id="photo-refiner", open=False) as accordion:
                pr_enabled = gr.Checkbox(value=False, label="Enable")
                gr.Markdown("━━━━━━")
                blur_intensity = gr.Slider(minimum=0, maximum=5, step=0.1, value=0, label="Blur")
                sharpen_intensity = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Sharpening")
                chromatic_aberration = gr.Slider(minimum=0, maximum=3, step=1, value=0, label="Chromatic Aberration")
                saturation_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Saturation")
                contrast_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Contrast")
                brightness_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Brightness")
                highlights_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Highlights")
                shadows_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Shadows")
                temperature_value = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Temperature")
                film_grain = gr.Checkbox(value=False, label="Filmic Grain")

                reset_button = gr.Button("Reset sliders")

                with gr.Accordion(label="Hints", elem_id="photo-refiner-hints", open=False) as accordion2:
                    gr.Markdown("### Wiki with examples: - https://github.com/Marc0ai/sd-webui-photo-refiner")
                
            def reset_sliders():
                return [0] * 10

            def update_title(pr_enabled):
                new_title = "Photo Refiner - Enabled" if pr_enabled else "Photo Refiner"
                return gr.update(label=new_title)

            def on_reset_button_click():
                return reset_sliders()

            reset_button.click(fn=on_reset_button_click, outputs=[
                blur_intensity,
                sharpen_intensity,
                chromatic_aberration,
                saturation_intensity,
                contrast_intensity,
                brightness_intensity,
                highlights_intensity,
                shadows_intensity,
                temperature_value
            ])

            pr_enabled.change(fn=update_title, inputs=pr_enabled, outputs=accordion)

        return [pr_enabled, temperature_value, blur_intensity, sharpen_intensity, chromatic_aberration, saturation_intensity, contrast_intensity, brightness_intensity, highlights_intensity, shadows_intensity, film_grain]

    def apply_effects(self, img, pr_enabled, temperature_value, blur, sharpen, ca, saturation, contrast, brightness, highlights, shadows, film_grain):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if temperature_value != 0:
            img_np = np.array(img).astype(np.float32) / 255.0
            img_np[..., 2] += temperature_value * 0.04
            img_np[..., 1] += temperature_value * 0.1 if temperature_value > 0 else img_np[..., 1] - temperature_value * 0.04
            img_np[..., 0] -= temperature_value * 0.04 if temperature_value < 0 else 0
            img_np = np.clip(img_np, 0, 1)
            img = Image.fromarray((img_np * 255).astype(np.uint8))

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
            grain_img = grain_img.filter(ImageFilter.GaussianBlur(radius=0.7))
            img = Image.blend(img.convert('RGB'), grain_img.convert('RGB'), alpha=0.025)

        return img

    def postprocess(self, p, processed, pr_enabled, temperature_value, blur_intensity, sharpen_intensity, chromatic_aberration, saturation_intensity, contrast_intensity, brightness_intensity, highlights_intensity, shadows_intensity, film_grain, *args):
        if pr_enabled:
            
            output_dir = "output/photo_refiner_outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            for i in range(len(processed.images)):
                if isinstance(processed.images[i], np.ndarray):
                    processed_image = Image.fromarray(processed.images[i])
                else:
                    processed_image = processed.images[i]

                processed_image = self.apply_effects(
                    processed_image,
                    pr_enabled,
                    temperature_value,
                    blur_intensity,
                    sharpen_intensity,
                    chromatic_aberration,
                    saturation_intensity,
                    contrast_intensity,
                    brightness_intensity,
                    highlights_intensity,
                    shadows_intensity,
                    film_grain
                )

                processed.images[i] = np.array(processed_image)

            for i, img_array in enumerate(processed.images):
                img = Image.fromarray(img_array)
                file_path = os.path.join(output_dir, f"output_image_{i}.png")
                img.save(file_path)
