import torch
import numpy as np
import gradio as gr
from PIL import Image
from network import ConditionalRealNVP
import os

MODEL_PATH = os.path.join("models", "conditional_robust_fcnn_100epochs.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on: {device}")

model = ConditionalRealNVP(dim=28*28, n_coupling_layers=8, hidden_dim=1024).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Load succesfully!")
else:
    print(f"Error: No file! {MODEL_PATH}")


def resize_image(numpy_img, target_size=(280, 280)):
    pil_img = Image.fromarray(numpy_img, mode='L')
    # resized_img = pil_img.resize(target_size, resample=Image.NEAREST)
    
    # smooth
    resized_img = pil_img.resize(target_size, resample=Image.BICUBIC)
    return resized_img



def predict_digit(digit):
    label = torch.tensor([int(digit)], device=device)

    z = torch.randn(1, 28*28, device=device)
    
    with torch.no_grad():
        x, _ = model.inverse(z, label)
        x_img = x.view(28, 28).cpu()
        # Clamp về [0, 1] rồi nhân 255 để ra chuẩn ảnh grayscale
        x_img = x_img.clamp(0, 1).numpy() * 255
        # Chuyển thành kiểu uint8 (số nguyên 0-255)
        x_img = x_img.astype(np.uint8)

        z_img = z.view(28, 28).cpu().numpy()
        # Normalize thủ công để hiển thị đẹp như utils.save_image(normalize=True)
        # Công thức: (z - min) / (max - min) * 255
        z_min, z_max = z_img.min(), z_img.max()
        z_img = (z_img - z_min) / (z_max - z_min) * 255
        z_img = z_img.astype(np.uint8)

        final_x = resize_image(x_img, target_size=(280, 280))
        final_z = resize_image(z_img, target_size=(280, 280))
    return final_x, final_z


if __name__ == "__main__":
    with gr.Blocks(title="Demo MNIST Generator") as demo:
        gr.Markdown("# 🎨 RealNVP Conditional Generator")
        gr.Markdown("Choose one number and model will generate corresponding image.")
        
        with gr.Row():
            with gr.Column():
                inp_slider = gr.Slider(minimum=0, maximum=9, step=1, value=0, label="Choose a digit to generate")
                btn_run = gr.Button("Generate 🚀", variant="primary")
            
            with gr.Column():
                out_image_x = gr.Image(label="Generated image(X)", height=280, width=280, image_mode='L')
                out_image_z = gr.Image(label="Noise input (Z)", height=280, width=280, image_mode='L')

        btn_run.click(fn=predict_digit, inputs=inp_slider, outputs=[out_image_x, out_image_z])

    print("Launching Gradio...")
    demo.launch(share=True)