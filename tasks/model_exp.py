import pandas as pd
from tqdm import tqdm
import json
import torch
from transformers import GenerationConfig, LlamaTokenizer
import os
import sys
sys.path.append('/home/weiguang/SaBART1/')
from transformers import LlamaTokenizer,LlamaConfig
import pytorch_lightning as pl ## have to import first to let torch has .fx
from transformers import LlamaForCausalLM, LlamaConfig
from src.models.chabot.chatbot_onehop_model import OneHopGNNBart
from peft import PeftModel, PeftConfig, LoraConfig, TaskType
from src.modules.chatbot.dataloader_onehop import (
    CommonGraphDataset
)
from torch.utils.data import DataLoader
from src.configuration.chatbot.config_args import parse_args_for_config
from tasks.chatbot.test import ChatbotTester


hparams = parse_args_for_config()
model = OneHopGNNBart(hparams)#.to_device("cuda:0")
layers = [i[0] for i in model.name_parameters()]
prob_layer = [i for i in layers if ]