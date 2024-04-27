import os
import modal
import torch
from container import stub

@stub.function(secret=modal.Secret.from_name("deeplsecret"))
def translate(sprache, textstring1, textstring2):
    if sprache == "Englisch":
        return textstring1, textstring2

    import deepl
    translator = deepl.Translator(os.environ["DeeplSecret"])
    if textstring1 != "":
        textstring1 = translator.translate_text(textstring1, target_lang="EN-US").text
    if textstring2 != "":
        textstring2 = translator.translate_text(textstring2, target_lang="EN-US").text
    return textstring1, textstring2


def generator(batch_size, rand_int):
    if rand_int == "":
        from random import randint
        rand_int = randint(1, 100000000)

    if type(rand_int) is str:
        from farmhash import Fingerprint32
        rand_int = Fingerprint32(rand_int)

    gen_list = []
    for i in range(batch_size):
        g_cpu = torch.Generator()
        g_cpu.manual_seed(rand_int+i)
        gen_list.append(g_cpu)
    return gen_list
