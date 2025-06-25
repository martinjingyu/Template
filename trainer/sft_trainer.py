import os

from transformers import HfArgumentParser, Trainer
from config import SFTConfig
import torch
import torch.nn.functional as F
def load_config():
    parser = HfArgumentParser(SFTConfig)
    config: SFTConfig = parser.parse_yaml_file("config/sft_model.yaml")[0]
    return config



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        )
        logits = outputs.get("logits")

        # shift for causal lm
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # loss 计算
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


def main():
    config = load_config()
    print(config.model_name_or_path)
    print(config.train_batch_size)
    
    
    

if __name__ == "__main__":
    
    
    main()