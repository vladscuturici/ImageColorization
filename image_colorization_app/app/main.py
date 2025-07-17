import os

import streamlit as st
import gradio as gr

from app.config import STYLE_IMAGE_PATH, MODEL_PATH
from app.inference import load_model, predict
from PIL import Image

model_dir = "model"
model_files = [
    os.path.join(model_dir, f) for f in os.listdir(model_dir)
    if f.endswith(".pth")
]
model_labels = [os.path.basename(f) for f in model_files]

model_cache = {}

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="gray",
    spacing_size="md",
    radius_size="lg",
    text_size="md"
)

def gradio_interface_with_gray(image, model):
    style_image = Image.open(STYLE_IMAGE_PATH).convert("RGB")
    gt_color = image.copy().convert("RGB")

    colorized, psnr, ssim = predict(model, image, style_image, gt_color)

    original = image.resize(colorized.size).convert("RGB")
    grayscale = image.convert("L").convert("RGB")

    if psnr is None and ssim is None:
        metric_vals = (
            f"**PSNR:** N/A  \n"
            f"**SSIM:** N/A"
        )
    else:
        metric_vals = (
            f"**PSNR:** {psnr:.2f} dB  \n"
            f"**SSIM:** {ssim:.4f}"
        )

    metric_expl = (
        "*Metrics not available for grayscale input*  \n"
        "*PSNR: higher is better, over 20 dB is good*  \n"
        "*SSIM: ranges from 0 to 1, closer to 1 is better; over 0.7 is good*"
    )

    return grayscale, colorized, (original, colorized), metric_vals, metric_expl

def wrapped_gradio_interface(image, model_path):
    if model_path not in model_cache:
        model_cache[model_path] = load_model(model_path)
    model = model_cache[model_path]
    return gradio_interface_with_gray(image, model)

def launch_app():
    st.title("Image Colorization App")
    st.markdown("Upload a grayscale image or try one of the examples below.")

    examples_path = "data/image_samples/"
    examples = None
    if os.path.exists(examples_path):
        example_files = [
            f for f in os.listdir(examples_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        examples = [[os.path.join(examples_path, f)] for f in example_files]

    default_model_path = MODEL_PATH if MODEL_PATH in model_files else model_files[0]

    with st.spinner("Launching interface..."):
        with gr.Blocks(theme=theme) as demo:
            gr.HTML("""
            <div style='text-align: center;'>
                <h1>Image Colorization using Deep Learning</h1>
                <hr style='margin-top: 1em;'>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=list(zip(model_labels, model_files)),
                        label="Select Model",
                        value=default_model_path
                    )

                    input_img = gr.Image(type="pil", label="Upload Image to Colorize")
                    submit_btn = gr.Button("Colorize")

                with gr.Column():
                    with gr.Row():
                        grayscale_out = gr.Image(type="pil", label="Grayscale", width=256, height=256)
                        output_img = gr.Image(type="pil", label="Colorized Output", width=256, height=256)

                    compare_slider = gr.ImageSlider(label="Slide to Compare", width=256, height=256)
                    metrics_md = gr.Markdown("")
                    metrics_expl = gr.Markdown("")

            gr.Examples(examples=examples, inputs=input_img)

            submit_btn.click(
                fn=wrapped_gradio_interface,
                inputs=[input_img, model_dropdown],
                outputs=[grayscale_out, output_img, compare_slider, metrics_md, metrics_expl]
            )

        demo.launch(inline=True, share=False)
