from modal import Stub, Image
from downloaders import download_models

"""
Effektiv ist diese Datei nur dafür da um den Container mit allem zu versorgen was er brauchen wird.
Werde hier nur dann etwas ändern, wenn ein Paket ein update braucht
"""
image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers~=0.19",
        "invisible_watermark~=0.1",
        "transformers~=4.31",
        "accelerate~=0.21",
        "safetensors~=0.3",
        "gradio",
        "compel",
        "deepl",
        "cityhash"
    )
    .run_function(download_models)
)

# Erstellt das Cloudobjekt mit dem dann weiter gearbeitet wird
# Der Name wird dann aber eh überschrieben in der wsgi app
stub = Stub("stable-diffusion-xl-working", image=image)
