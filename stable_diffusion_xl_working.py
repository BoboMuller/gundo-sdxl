from fastapi import FastAPI
from modal import asgi_app, method
from container import stub
from tools import translate, generator

web_app = FastAPI()
TIMEOUT = 90
GPU = "A100"

# Ressourcenparameter, es sind auch mehrere GPU möglich
@stub.cls(gpu=GPU, container_idle_timeout=TIMEOUT)
class Model():
    def __init__(self, sched, LoRA):
        import torch
        from diffusers import DiffusionPipeline
        from compel import Compel, ReturnedEmbeddingsType

        # Muss noch vor dem Modell geladen werden, weil es davon abhängig ist
        self.scheduler = self.get_right_sched(sched)

        # Einstellungen die Modell und Refiner betreffen
        # 16 bit VAE müsste dann auch hier rein. Müsste mal getestet werden
        # 16 bit VAE evtl. zu einer nutzerentscheidung machen? Custom VAE support?
        load_options = dict(
            scheduler=self.scheduler,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Lädt das Basismodell
        # Die Pipeline sollte evtl. auseinandergenommen werden für mehr Konfigurationsmöglichkeiten
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Lädt den Refiner
        # TODO: Option zum überspringen ist für einige LoRA teilweise nötig (Pixel)
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

        # LoRA werden zwischen die Schichten geschoben, daher erst jetzt möglich
        self.apply_lora(LoRA)

        # Compbel übernimmt das Umwandeln von Token/Worte in Embeddings um Gewichte setzen zu können
        # Dude with a large hat -> Dude with a large++ hat -> Dude with a large(1.5) hat
        # 2.0 bedeutet aber nicht doppelt so wichtig. Das ist fast schon ein subjektiver Parameter
        self.compel = Compel(tokenizer=[self.base.tokenizer, self.base.tokenizer_2],
                        text_encoder=[self.base.text_encoder, self.base.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True])


    def apply_lora(self, name):
        """
        Schiebt das LoRA ins Modell
        """
        from safetensors.torch import load_file
        cache_path = "/vol/cache"

        if name == "Nichts":
            return
        else:
            lora_path = f"{cache_path}/LoRA/{name}.safetensors"
        lora_state_dict = load_file(lora_path)
        self.base.load_lora_weights(lora_state_dict)


    def get_right_sched(self, name):
        """
        Abhängig vom Parameter wird der richtige Scheduler gewählt, geladen und zurückgegeben
        """
        from diffusers import DPMSolverMultistepScheduler, DEISMultistepScheduler, CMStochasticIterativeScheduler, EulerAncestralDiscreteScheduler
        cache_path = "/vol/cache"

        # Um Code zu reduzieren ein Key Value storage
        # Key: Namen der Methoden
        # Value: Tuple aus dem Objekt und einem weiteren dictionary mit den dazugehörend benötigten Parametern
        schedulers = {
            "DPM++2M": (DPMSolverMultistepScheduler, {"solver_type": "midpoint", "algorithm_type": "dpmsolver++"}),
            "DPM++2M-Karras": (DPMSolverMultistepScheduler, {"solver_type": "midpoint", "algorithm_type": "dpmsolver++", "use_karras_sigmas": True}),
            "DEIS": (DEISMultistepScheduler, {"algorithm_type": "deis"}),
            "CMS_TEST": (CMStochasticIterativeScheduler, {}),
            "Euler-a": (EulerAncestralDiscreteScheduler, {}),
        }

        scheduler_class, params = schedulers[name]
        # Hier werden dann auch alle anderen Parameter eingefügt die immer nötig sind, oder zumindest ignoriert werden
        return scheduler_class.from_pretrained(cache_path, subfolder="scheduler", solver_order=2, prediction_type="epsilon", device_map="auto", **params)


    @method()
    def inference(self, prompt, negative_prompt, batch_size, steps, fraq, guidance, rand_val, width=1024, height=1024):
        conditioning, pooled = self.compel(prompt)
        conditioning1, pooled1 = self.compel(negative_prompt)
        gen_list = generator(batch_size, rand_val)

        image = self.base(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=conditioning1,
            negative_pooled_prompt_embeds=pooled1,
            generator=gen_list,
            num_inference_steps=steps,
            denoising_end=fraq,
            num_images_per_prompt=batch_size,
            guidance_scale=guidance,
            height=height,
            width=width,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=gen_list,
            num_inference_steps=steps,
            denoising_start=fraq,
            num_images_per_prompt=batch_size,
            guidance_scale=guidance,
            image=image,
        ).images
        return image


@stub.function()
@asgi_app(label="sdxl-gundo")
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app
    import math
    from PIL import Image

    def return_images(prompt, negative_prompt, batch_size, steps, fraq, guidance, dpm, lora, sprache, rand_val, resolution):
        width, height = resolution.split("x")
        width, height = int(width), int(height)
        model = Model(dpm, lora)
        t_prompt, t_negative_prompt = translate.remote(sprache, prompt, negative_prompt)

        #generated = model.inference.remote(t_prompt, t_negative_prompt, batch_size, steps, fraq, guidance, rand_val, width, height)
        full_arr = math.floor(batch_size/4)
        remain = batch_size%4
        tmp = []
        for i in range(full_arr):
            generated = model.inference.remote(t_prompt, t_negative_prompt, 4, steps, fraq, guidance, rand_val, width, height)
            tmp = tmp + generated
        if remain != 0:
            generated = model.inference.remote(t_prompt, t_negative_prompt, remain, steps, fraq, guidance, rand_val, width, height)
            tmp = tmp + generated

        #for i in range(max_batch - batch_size):
        #    generated.append(Image.new('RGB', (1, 1), (255, 255, 255)))
        return tmp #generated

    # alt, evtl. löschen
    def change_vis(batch_size):
        show = [gr.Image.update(visible=True) for i in range(batch_size)]
        hide = [gr.Image.update(visible=False, value=None) for i in range(max_batch - batch_size)]
        return show + hide


    with gr.Blocks() as interface:
        with gr.Group():
            prompt = gr.Textbox(label="Bildbeschreibung", info="Mit + und - am Ende lässt sich die Wichtigkeit ändern", placeholder="schönes haus, winter, schneemann, alt, gemälde, teuer, hochwertig, perfekt, ölfarben")
            negative_prompt = gr.Textbox(label="Gegenteilige Bildbeschreibung")
            sprache = gr.Radio(["Englisch", "Deutsch, Spanisch, Französich uvm."], value="Englisch", label="Sprache deiner Eingabe", info="Bei einer Mischung bitte Deutsch wählen")
        with gr.Row():
            steps = gr.Slider(minimum=10, maximum=70, step=1.0, value=30, label='Generierungsschritte', info='Streng genommen "Entrauschungsschritte" und übertreiben kann zu komischen Bildern führen')
            fraq = gr.Slider(minimum = 0.5, maximum=1.0, step=0.05, value=0.8, label="Generierung/Verbesserung", info="Verhältnis von Generierungsschritten und Verbesserungsschritten")
            guidance = gr.Slider(minimum=5, maximum=13, step=0.5, value=7.5, label="Einlenkungswert", info="Je höher desto eher wird genau gemacht was man will")
        dpm = gr.Radio(["DPM++2M-Karras", "DPM++2M", "DEIS", "Euler-a", "CMS_TEST"], value="DPM++2M-Karras", label="Vorhersage-Algorithmus", info="DPM++ kann schon mit 25 Schritten gute Ergebnisse haben. Andere brauchen etwa 50. Einfach mal rumprobieren")
        lora = gr.Radio(["Nichts", "Pixelart", "PS1", "Sticker" ], value="Nichts", label="Verbesserung", info="Nachbesserungen")

        with gr.Row():
            max_batch = 16
            batch_size = gr.Slider(minimum=1, maximum=max_batch, step=1.0, label="Anzahl der Bilder")
            gen = gr.Textbox(label="Zufallsgenerator", placeholder="Zahl oder Buchstaben um ein konstantes Ergebnis zu bekommen")
        resolution = gr.Dropdown(["1024x1024", "1152x896", "1216x832", "1336x768", "1536x640", "896x1152", "832x1216", "768x1336", "640x1536"], label="Auflösung", value="1024x1024")
        gen_btn = gr.Button("Generieren")

        #alt, evtl. löschen
        #images = []
        #images.append(gr.Image(interactive=False, visible=True))
        #for i in range(max_batch-1):
        #    img = gr.Image(interactive=False, visible=False)
        #    images.append(img)
        #alt, löschen

        #neu
        gallery = gr.Gallery(label="Bildergallerie", object_fit="contain", height="auto", columns=[3])
        #neu

        #batch_size.input(fn=change_vis, inputs=batch_size, outputs=images)
        gen_btn.click(fn=return_images, inputs=[prompt, negative_prompt, batch_size, steps, fraq, guidance, dpm, lora, sprache, gen, resolution], outputs=gallery) #images)

    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )
