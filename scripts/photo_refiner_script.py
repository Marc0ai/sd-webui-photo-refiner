import gradio as gr
import os
import modules.scripts as scripts
from modules import ui_components, shared, util, paths_internal
from modules import ui_components, shared, util, paths_internal, scripts_postprocessing
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageChops, ImageFilter
from datetime import datetime

def apply_effects(img, 
    denoise_intensity,
    temperature_value,
    blur_intensity,
    sharpen_intensity,
    chromatic_aberration,
    saturation_intensity,
    contrast_intensity,
    brightness_intensity,
    highlights_intensity,
    shadows_intensity,
    film_grain,
    sepia_filter
):
        
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        
    if img.mode != 'RGB':
        img = img.convert('RGB')

    if denoise_intensity > 0:
        img_np = np.array(img)
        img_np = cv2.fastNlMeansDenoisingColored(img_np, None, denoise_intensity, denoise_intensity, 5, 18)
        img = Image.fromarray(img_np)   

    if temperature_value != 0:
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np[..., 2] += temperature_value * 0.04
        img_np[..., 1] += temperature_value * 0.1 if temperature_value > 0 else img_np[..., 1] - temperature_value * 0.04
        img_np[..., 0] -= temperature_value * 0.04 if temperature_value < 0 else 0
        img_np = np.clip(img_np, 0, 1)
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        
    if blur_intensity > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_intensity))
        
    if sharpen_intensity > 0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpen_intensity)
        
    if chromatic_aberration > 0:
        r, g, b = img.split()
        r = ImageChops.offset(r, chromatic_aberration, 0)
        b = ImageChops.offset(b, -chromatic_aberration, 0)
        img = Image.merge('RGB', (r, g, b))
        
    if saturation_intensity != 0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_intensity / 5.0 + 1)
        
    if contrast_intensity != 0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_intensity / 5.0 + 1)
        
    if brightness_intensity != 0:
        img = Image.fromarray(np.clip(np.array(img) * (brightness_intensity / 5.0 + 1), 0, 255).astype(np.uint8))
        
    if highlights_intensity != 0 or shadows_intensity != 0:
        img_np = np.array(img).astype(np.float32) / 255.0
        if shadows_intensity != 0:
            gamma = 1.0 / (shadows_intensity / 5.0 + 1) if shadows_intensity > 0 else 1.0 + abs(shadows_intensity / 5.0)
            img_np = np.power(img_np, gamma)
        if highlights_intensity != 0:
            highlights_scale = highlights_intensity / 5.0 + 1
            img_np[img_np > 0.5] = 0.5 + (img_np[img_np > 0.5] - 0.5) * highlights_scale
        img = Image.fromarray(np.clip(img_np * 255, 0, 255).astype(np.uint8))
        
    if film_grain:
        grain = np.random.normal(0, 1, (img.height, img.width))
        grain = np.clip(grain * 255, 0, 255).astype(np.uint8)
        grain_img = Image.fromarray(grain).convert('L')
        grain_img = grain_img.resize((img.width, img.height), Image.NEAREST)
        grain_img = grain_img.filter(ImageFilter.GaussianBlur(radius=0.7))
        img = Image.blend(img.convert('RGB'), grain_img.convert('RGB'), alpha=0.025)
        
    if sepia_filter:
        img_np = np.array(img).astype(np.float32) / 255.0
        sepia_effect = np.array(
            [[0.393, 0.769, 0.189],
             [0.349, 0.686, 0.168],
             [0.272, 0.534, 0.131]]
        )
        img_np = img_np.dot(sepia_effect.T)
        img_np = np.clip(img_np, 0, 1)
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        
    return img
    
class Script(scripts.Script):
    
    def title(self, enabled=False):
        return "Photo Refiner 2.0"
        
    def show(self, is_img2img):
        return scripts.AlwaysVisible
        
    def ui(self, is_img2img):
        with ui_components.InputAccordion(False, label=self.title()) as pr_enabled:
            with gr.Row():
                blur_intensity = gr.Slider(minimum=0, maximum=5, step=0.1, value=0, label="Blur")
                sharpen_intensity = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Sharpening")  
                denoise_intensity = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Denoise")
            with gr.Row():    
                chromatic_aberration = gr.Slider(minimum=0, maximum=5, step=1, value=0, label="Chromatic Aberration")
                saturation_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Saturation")
                contrast_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Contrast")
            with gr.Row():
                brightness_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Brightness")
                highlights_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Highlights")
                shadows_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Shadows")
            with gr.Row():    
                temperature_value = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Temperature")
                sepia_filter = gr.Checkbox(value=False, label="Sepia Efect")
                film_grain = gr.Checkbox(value=False, label="Filmic Grain")

            reset_button = gr.Button("Reset sliders")

        def reset_sliders():
            return [0] * 11
            
        def on_reset_button_click():
            return reset_sliders()
            
        reset_button.click(fn=on_reset_button_click, outputs=[
            blur_intensity,
            sharpen_intensity,
            denoise_intensity,
            chromatic_aberration,
            saturation_intensity,
            contrast_intensity,
            brightness_intensity,
            highlights_intensity,
            shadows_intensity,
            temperature_value
        ])
        return [pr_enabled,
            denoise_intensity,
            temperature_value,
            blur_intensity,
            sharpen_intensity,
            chromatic_aberration,
            saturation_intensity,
            contrast_intensity,
            brightness_intensity,
            highlights_intensity,
            shadows_intensity,
            film_grain,
            sepia_filter
        ]

    def postprocess(self, p, processed,
        pr_enabled,
        denoise_intensity,
        temperature_value,
        blur_intensity,
        sharpen_intensity,
        chromatic_aberration,
        saturation_intensity,
        contrast_intensity,
        brightness_intensity,
        highlights_intensity,
        shadows_intensity,
        film_grain,
        sepia_filter,
        *args
    ):
            
        if pr_enabled:

            output_dir = "output/photo_refiner_outputs"
            os.makedirs(output_dir, exist_ok=True)

            for i in range(len(processed.images)):
                if isinstance(processed.images[i], np.ndarray):
                    processed_image = Image.fromarray(processed.images[i])
                else:
                    processed_image = processed.images[i]
    
                processed_image = apply_effects(
                    processed_image,
                    denoise_intensity,
                    temperature_value,
                    blur_intensity,
                    sharpen_intensity,
                    chromatic_aberration,
                    saturation_intensity,
                    contrast_intensity,
                    brightness_intensity,
                    highlights_intensity,
                    shadows_intensity,
                    film_grain,
                    sepia_filter
                )
    
                processed.images[i] = np.array(processed_image)
    
            for i, img_array in enumerate(processed.images):
                img = Image.fromarray(img_array)
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
                file_path = os.path.join(output_dir, f"{timestamp}.png")
                img.save(file_path)


class PhotoRefinerPP(scripts_postprocessing.ScriptPostprocessing):
    name = "Photo Refiner 2.0"
    order = 90000

    def ui(self):
        with ui_components.InputAccordion(False, label=self.name) as pr_enabled:
            with gr.Row():
                blur_intensity = gr.Slider(minimum=0, maximum=5, step=0.1, value=0, label="Blur")
                sharpen_intensity = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Sharpening")  
                denoise_intensity = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Denoise")
            with gr.Row():    
                chromatic_aberration = gr.Slider(minimum=0, maximum=5, step=1, value=0, label="Chromatic Aberration")
                saturation_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Saturation")
                contrast_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Contrast")
            with gr.Row():
                brightness_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Brightness")
                highlights_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Highlights")
                shadows_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Shadows")
            with gr.Row():    
                temperature_value = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Temperature")
                sepia_filter = gr.Checkbox(value=False, label="Sepia Efect")
                film_grain = gr.Checkbox(value=False, label="Filmic Grain")

            reset_button = gr.Button("Reset sliders")

        def reset_sliders():
            return [0] * 11

        def on_reset_button_click():
            return reset_sliders()

        reset_button.click(fn=on_reset_button_click, outputs=[
            blur_intensity,
            sharpen_intensity,
            denoise_intensity,
            chromatic_aberration,
            saturation_intensity,
            contrast_intensity,
            brightness_intensity,
            highlights_intensity,
            shadows_intensity,
            temperature_value
        ])

        return {
            'pr_enabled': pr_enabled,
            'temperature_value': temperature_value,
            'blur_intensity': blur_intensity,
            'sharpen_intensity': sharpen_intensity,
            'denoise_intensity': denoise_intensity,
            'chromatic_aberration': chromatic_aberration,
            'saturation_intensity': saturation_intensity,
            'contrast_intensity': contrast_intensity,
            'brightness_intensity': brightness_intensity,
            'highlights_intensity': highlights_intensity,
            'shadows_intensity': shadows_intensity,
            'film_grain': film_grain,
            'sepia_filter': sepia_filter,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if not args.pop('pr_enabled', False):
            return

        pp.image = apply_effects(img=pp.image, **args)