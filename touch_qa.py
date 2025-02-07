import ImageBind.data as data
import llama
import numpy as np
import torch
from scipy.interpolate import interp1d
import os

llama_dir = "./llama_ori"


    
# checkpoint will be automatically downloaded
model = llama.load("7B", llama_dir, llama_type="7B_chinese", knn=True)
model.eval()


inputs = {}


touch = data.load_and_transform_vision_data(["./touch_1/0000016865.jpg"], device="cuda")
inputs['Touch'] = [touch, 1]


results = model.generate(
    inputs,
    [llama.format_prompt("You will be presented with an touch image from a object/surface. Can you describe the touch feeling and the texture?")],
    max_gen_len=256
)
result = results[0].strip()
print(result)