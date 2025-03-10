from transformers import pipeline
import torch

# check if GPU is avaliable
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = pipeline(task="summarization", model="facebook/bart-large-cnn")
response = model("text to summarize")
print(response)