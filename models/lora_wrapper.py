from peft import get_peft_model, LoraConfig, TaskType

def add_lora(model, r=8, alpha=32, dropout=0.05):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none"
    )
    return get_peft_model(model, config)