import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
from scipy.ndimage import gaussian_filter

import cv2
import dlib
import numpy as np
from PIL import Image, ImageEnhance, ImageChops, ImageFilter, ImageDraw
    
detector = dlib.get_frontal_face_detector()

class Script(scripts.Script):
    
    def title(self, enabled=False):
        if enabled:
            return "Photo Refiner - Enabled - TEST"
        else:
            return "Photo Refiner - TEST"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
                        
        with gr.Blocks() as demo:  
            with gr.Accordion(label=self.title(), elem_id="photo-refinerT", open=False) as accordion:
                pr_enabled = gr.Checkbox(value=False, label="Enable")
                blur_intensity = gr.Slider(minimum=0, maximum=5, step=0.1, value=0, label="Blur")
                sharpen_intensity = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Sharpening")
                chromatic_aberration = gr.Slider(minimum=0, maximum=3, step=1, value=0, label="Chromatic Aberration")
                saturation_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Saturation")
                contrast_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Contrast")
                brightness_intensity = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Brightness")
                highlights_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Highlights")
                shadows_intensity = gr.Slider(minimum=-10, maximum=10, step=0.1, value=0, label="Shadows")
                temperature_value = gr.Slider(minimum=-5, maximum=5, step=0.1, value=0, label="Temperature")
                red_intensity = gr.Slider(minimum=0, maximum=2, step=0.1, value=1, label="Red")
                green_intensity = gr.Slider(minimum=0, maximum=2, step=0.1, value=1, label="Green")
                blue_intensity = gr.Slider(minimum=0, maximum=2, step=0.1, value=1, label="Blue")
                film_grain = gr.Checkbox(value=False, label="Filmic Grain")
                face_sharp_intensity = gr.Slider(minimum=0, maximum=10, step=0.1, value=0, label="Face Enhancer (Experimental)")
            
            def update_title(pr_enabled):
                new_title = "Photo Refiner - Enabled - TEST" if pr_enabled else "Photo Refiner - TEST"
                return gr.update(label=new_title)
            
            pr_enabled.change(fn=update_title, inputs=pr_enabled, outputs=accordion) 
            
        return [pr_enabled, face_sharp_intensity, red_intensity, green_intensity, blue_intensity, temperature_value, blur_intensity, sharpen_intensity, chromatic_aberration, saturation_intensity, contrast_intensity, brightness_intensity, highlights_intensity, shadows_intensity, film_grain]

    def apply_effects(self, im, pr_enabled, face_sharp_intensity, red_intensity, green_intensity, blue_intensity, temperature_value, blur, sharpen, ca, saturation, contrast, brightness, highlights, shadows, film_grain):
       
        if isinstance(im, np.ndarray):
            img = Image.fromarray(im)
        else:
            img = im
             
        if face_sharp_intensity > 0:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            faces = detector(gray)
    
            mask = Image.new("L", img.size, 0)
    
            for face in faces:
                left, top, right, bottom = (face.left(), face.top(), face.right(), face.bottom())
                mask_ellipse = Image.new("L", img.size, 0)
                mask_draw = ImageDraw.Draw(mask_ellipse)
                mask_draw.ellipse([left, top, right, bottom], fill=255)
                
                mask_ellipse = mask_ellipse.filter(ImageFilter.GaussianBlur(40))
                mask = ImageChops.add(mask, mask_ellipse)
    
            enhancer = ImageEnhance.Sharpness(img)
            sharp_img = enhancer.enhance(face_sharp_intensity)
            
            img = Image.composite(sharp_img, img, mask)    
            
        if temperature_value != 0:
            img_np = np.array(img).astype(np.float32) / 255.0
          
            if temperature_value > 0:
                img_np[..., 2] += temperature_value * 0.04
                img_np[..., 1] += temperature_value * 0.04
            else:
                img_np[..., 2] += temperature_value * 0.04
                img_np[..., 0] -= temperature_value * 0.04
                
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
            
        if red_intensity != 0 or green_intensity != 0 or blue_intensity != 0:
            pixels = np.array(img)
            
            mask_red = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.float32)
            mask_green = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.float32)
            mask_blue = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.float32)
            
            r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
            max_color = np.maximum(r, np.maximum(g, b))
            
            mask_red = np.where((r > g + 1.5) & (r > b + 1.5), (r - np.minimum(g, b)) / max_color, 0)
            mask_green = np.where((g > r + 1.5) & (g > b + 1.5), (g - np.minimum(r, b)) / max_color, 0)
            mask_blue = np.where((b > r + 1.5) & (b > g + 1.5), (b - np.minimum(r, g)) / max_color, 0)
            
            mask_red = gaussian_filter(mask_red, sigma=10)
            mask_green = gaussian_filter(mask_green, sigma=10)
            mask_blue = gaussian_filter(mask_blue, sigma=10)
            
            mask_red = np.clip(mask_red, 0, 1)
            mask_green = np.clip(mask_green, 0, 1)
            mask_blue = np.clip(mask_blue, 0, 1)
            
            r = np.clip(r * (1 + (red_intensity - 1) * mask_red), 0, 255)
            g = np.clip(g * (1 + (green_intensity - 1) * mask_green), 0, 255)
            b = np.clip(b * (1 + (blue_intensity - 1) * mask_blue), 0, 255)
            
            adjusted_pixels = np.stack([r, g, b], axis=-1).astype(np.uint8)
            img = Image.fromarray(adjusted_pixels)
    
        return img

    def postprocess(self, p, processed, pr_enabled, face_sharp_intensity, red_intensity, green_intensity, blue_intensity, temperature_value, blur_intensity, sharpen_intensity, chromatic_aberration, saturation_intensity, contrast_intensity, brightness_intensity, highlights_intensity, shadows_intensity, film_grain, *args):

        if pr_enabled:

            for i in range(len(processed.images)):
                if state.interrupted:
                    break
           
                if isinstance(processed.images[i], np.ndarray):
                    processed_image = Image.fromarray(processed.images[i])
                else:
                    processed_image = processed.images[i]
           
           
                processed_image = self.apply_effects(
                    processed_image,
                    pr_enabled,
                    face_sharp_intensity, 
                    red_intensity,
                    green_intensity,
                    blue_intensity,
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
                
                processed.images[0] = np.array(processed_image)
