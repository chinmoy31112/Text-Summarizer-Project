from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        """Takes raw text and return a generated summary using direct model generation."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

        # Move to GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Prepare payload
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        # Generation configuration
        gen_kwargs = {
            "length_penalty": 2.0,
            "num_beams": 4,
            "max_length": 128,
            "min_length": 30
        }

        # Generate summary
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
