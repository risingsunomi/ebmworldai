"""
VisionLLM
Using the AlexNet and Energy readings to pass to an llm
for generative descriptions of what it sees pooled from 
finetuning/traing on video energy readings and text video subtitles
"""
from cv2.typing import UMat
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM

class VisionLLM:
    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained("path/to/pretrained/llama/tokenizer")
        self.llama_model = LlamaForCausalLM.from_pretrained("path/to/pretrained/llama/model")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llama_model.to(device)

    def generate_description(
            self,
            subtitle, 
            contour_energy, 
            llama_model, 
            tokenizer, 
            device
        ):

        with torch.no_grad():
            input_ids = tokenizer(subtitle, return_tensors="pt").to(device)
            llama_outputs = llama_model.generate(
                input_ids=input_ids,
                additional_input=contour_energy,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            generated_description = tokenizer.decode(llama_outputs[0], skip_special_tokens=True)
        return generated_description