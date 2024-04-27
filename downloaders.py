def pre_download_lora():
    """
    Lädt die ganzen LoRA runter die in der link Liste sind.
    Dieser Methode ist egal um welches Modell es sich handelt
    """
    from pathlib import Path
    import shutil
    import requests

    cache_path = "/vol/cache"

    # Hier den LoRA Link einfügen und einen passenden Namen.
    # Beim Link drauf achten dass type und format gesetzt sind
    # Generell muss der Name identisch sein zur Bezeichnung des Knopfs im UI
    links = ['https://civitai.com/api/download/models/135931?type=Model&format=SafeTensor',
             'https://civitai.com/api/download/models/147912?type=Model&format=SafeTensor',
             'https://civitai.com/api/download/models/136078?type=Model&format=SafeTensor']
    names = ["Pixelart", "PS1", "Sticker"]

    for link, name in zip(links, names):
        url = link
        response = requests.get(url, stream=True)
        filepath = Path(f"{cache_path}/LoRA/{name}.safetensors")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w+b") as out_file:
            shutil.copyfileobj(response.raw, out_file)

def download_sched_xl(scheduler):
    cache_path = "/vol/cache"
    options_xl = dict(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler",
        cache_dir=cache_path
        )

    loaded_sched = scheduler.from_pretrained(**options_xl)
    loaded_sched.save_pretrained(cache_path, safe_serialization=True)

def download_models():
    """
    Downloadmethode für die Modelle an sich und für die Scheduler
    Weitere Modelle können hier hinzugefügt werden
    Beachte, dass Scheduler abhängig sind vom Modell
    """
    from huggingface_hub import snapshot_download
    import diffusers
    from multiprocessing import Pool, cpu_count

    sched_list = [diffusers.DPMSolverMultistepScheduler,
                  diffusers.DEISMultistepScheduler,
                  diffusers.CMStochasticIterativeScheduler,
                  diffusers.EulerAncestralDiscreteScheduler]

    pool = Pool(cpu_count())
    print(cpu_count())
    pool.map(download_sched_xl, sched_list)
    pool.close()
    pool.join()

    ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
    )
    snapshot_download(
        "stabilityai/stable-diffusion-xl-refiner-1.0", ignore_patterns=ignore
    )
    #snapshot_download(
    #    "diffusers/controlnet-canny-sdxl-1.0", ignore_patterns=ignore
    #)
    snapshot_download(
        "madebyollin/sdxl-vae-fp16-fix", ignore_patterns=ignore
    )


    #scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(**options_xl)
    #scheduler.save_pretrained(cache_path, safe_serialization=True)

    #scheduler1 = diffusers.DEISMultistepScheduler.from_pretrained(**options_xl)
    #scheduler1.save_pretrained(cache_path, safe_serialization=True)

    #scheduler2 = diffusers.CMStochasticIterativeScheduler.from_pretrained(**options_xl)
    #scheduler2.save_pretrained(cache_path, safe_serialization=True)

    #scheduler3 = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(**options_xl)
    #scheduler3.save_pretrained(cache_path, safe_serialization=True)

    pre_download_lora()