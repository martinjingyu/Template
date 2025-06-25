from transformers import AutoModelForCausalLM

def load_base_model(model_name: str, freeze_layers=True):
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if freeze_layers:
        for name, param in model.named_parameters():
            if "transformer.h." in name and int(name.split(".")[2]) < 20:
                param.requires_grad = False
    return model
