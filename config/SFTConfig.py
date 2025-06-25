from dataclasses import dataclass, field

@dataclass
class SFTConfig:
    
    
    
    train_batch_size: int = field(default=8)
    eval_batch_size: int = field(default=4)
    learning_rate: float = field(default=2e-5)
    num_train_epochs: int = field(default=3)
    output_dir: str = field(default="./output")
    model_name_or_path: str = field(default="gpt2")
    logging_dir: str = field(default="./logs")
    
    