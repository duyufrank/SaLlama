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
#path = "/home/weiguang/SaBART1/result_bos_3epochs.json"
# start_index = -1
# ## read from record file to know when we should start
# if os.path.exists(path):
# 	with open(path,"r") as f:
# 		for line in f:
# 			line_data = json.loads(line.strip())
# 		start_index = int(list(line_data.keys())[0]) if line_data else -1
# print('The inference will start from the index '+str(start_index+1)+' of the test data.')
# breakpoint()
# load the original LlaMa
# MODEL_NAME = "/data/weiguang/Llama2-Chinese-7b-Chat"
# tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
# model = LlamaForCausalLM.from_pretrained(MODEL_NAME).to('cuda:2')
# best_model = OneHopGNNBart
# # load our fine-tuned Llama model
# best_model =
# breakpoint()
# type(best_model.model): <class 'peft.peft_model.PeftModelForCausalLM'>
# type(best_model.model.base_model): <class 'peft.tuners.lora.model.LoraModel'>
# type(best_model.model.base_model.model): <class 'src.models.transformers.modeling_llama.LlamaForCausalLM'>
# tokenizer = best_model.tokenizer
# model = best_model.model.to('cuda:2')
# model = model.merge_and_unload() ## merge the lora adapters with base model to use it as a standalone base model

# config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.1)
# model = PeftModel.from_pretrained(model, peft_model_id).to("cuda:0")
# generation_config = GenerationConfig(max_new_tokens=512)

#data = pd.read_csv("/home/weiguang/SaBART/matinf_1.0_encrypted/testset.txt")
hparams = parse_args_for_config()
# root = '/home/weiguang/SaBART1/'
# data_dir = root + 'datasets/matinf_1.0_encrypted'
# model_name_or_path = root + 'data/weiguang/Llama2-Chinese-7b-Chat'
batch_size=1
tester = ChatbotTester(hparams)# to read model --> OneHopGNNBart
Model = tester.model.to('cuda:0')
Model.model.resize_token_embeddings(new_num_tokens=len(Model.tokenizer))
tokenizer = Model.tokenizer
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
#dataloader = Model.test_dataloader()
dataset = CommonGraphDataset(data_dir=hparams.data_dir, hparams=hparams, batch_size=batch_size,\
                                 tokenizor= tokenizer,\
                                 embedding_size=LlamaConfig.from_pretrained(hparams.model_name_or_path), data_name='test')

with open("model_structure.txt", "w", encoding="utf-8") as f:
    f.write(str(Model))
breakpoint()
def print_structure(module, indent=0, file=None):
    if file is None:
        file = open("model_structure.txt", "w", encoding="utf-8")  # Open file for writing
    for name, child in Model.named_parameters(): print(name)
        file.write('\t' * indent + name + '\n')
        print_structure(child, indent + 1, file)
    file.close()  # Close the file after writing is done

# Call the function to print structure
print_structure(Model)
breakpoint()
dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            num_workers=hparams.num_workers,
        )

for idx,batch in enumerate(dataloader):

    src_ids = batch['src_ids'].to(Model.device)
    tgt_ids = batch['tgt_ids'].to(Model.device)
    encoder_attention = batch['encoder_attention'].to(Model.device)
    decoder_attention = batch['decoder_attention'].to(Model.device)
    one_hop_graph = [batch['graph'][0].to(Model.device)]
    

    
    # csk_ids = []
    # for _one_hop_graph in one_hop_graph:
    #         _csk_ids = _one_hop_graph.ndata['h'][:, [0, 3]]
    #         _csk_ids = set([int(ids) for tri in _csk_ids for ids in tri])
    #         csk_ids.append(_csk_ids)

    # transformers.generation_utils
    extra_params = {}
    if hparams.num_beam_groups > 1:
        extra_params["num_beam_groups"] = hparams.num_beam_groups
        extra_params["diversity_penalty"] = hparams.diversity_penalty
    if Model.eval_beams >= 1:
        extra_params["num_beams"] = Model.eval_beams
    if hparams.repetition_penalty > 0:
        extra_params["repetition_penalty"] = hparams.repetition_penalty

    generated_ids = Model.model.generate(
        input_ids=torch.tensor([tokenizer.bos_token_id for _
                                in range(len(tgt_ids))])[:, None].to(src_ids.device),
        src_ids=src_ids,
        encoder_attention=encoder_attention,
        one_hop_graph = one_hop_graph,
        decoder_start_token_id=Model.decoder_start_token_id,
        max_length=hparams.gen_max_len,
        min_length=hparams.gen_min_len,
        top_p=Model.top_p if Model.use_top_p else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **extra_params
    )
    preds = Model.gen_ids_to_clean_text(generated_ids)
    with open('filename', 'w', encoding='utf8') as json_file:
        json.dump(preds, json_file, ensure_ascii=False)
    breakpoint()
#proc steps
def get_dataset(data_name: str, batch_size: int) -> CommonGraphDataset:
        dataset = CommonGraphDataset(
            data_dir=self.hparams.data_dir,
            hparams=self.hparams,
            batch_size=batch_size,
            tokenizor=self.tokenizer,
            embedding_size=self.config.hidden_size,
            data_name=data_name,
        )
        self.model.model.resize_token_embeddings(new_num_tokens=len(self.model.tokenizer))
        return dataset
def get_dataloader( data_name: str,  batch_size: int, shuffle: bool = False):
        dataset = get_dataset(data_name, batch_size)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )
data = get_dataloader('testset.txt',2)

#may use f:__get_item__ in dataloader_onehop.py
r = []
for idx,batch in enumerate(data):

    src_ids = batch['src_ids']
    tgt_ids = batch['tgt_ids']
    encoder_attention = batch['encoder_attention']
    decoder_attention = batch['decoder_attention']
    one_hop_graph = batch['graph']
    generated_ids = best_model.generate(
            input_ids=torch.tensor([self.tokenizer.bos_token_id for _
                                    in range(len(tgt_ids))])[:, None].to(src_ids.device),
            src_ids=src_ids,
            encoder_attention=encoder_attention,
            one_hop_graph = one_hop_graph,
            decoder_start_token_id=self.decoder_start_token_id,
            max_length=self.hparams.gen_max_len,
            min_length=self.hparams.gen_min_len,
            top_p=self.top_p if self.use_top_p else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **extra_params
        )
    r_batch = []
    for i in range(batch.shape[0]):
        templist = ''.join([tokenizer(id) for id in generated_ids[i]])
        r_batch.append(templist)
    r.append(r_batch)
r = [i for j in r for i in j]
with open(path, "a", encoding='utf8') as f:
    for i in r:
        json.dump(i,f,ensure_ascii=False)
        f.write('\n')



batch_size = 8
for i in tqdm(range(start_index+1, len(data), batch_size)):
    prompt = []
    for j in range(batch_size):
        if i+j < len(data):
            prompt.append(f"<s>问题: "+data['question'][i+j] + data['description'][i+j]+"\n\n</s><s>回答: ")
    encoding = tokenizer(prompt, padding=True, max_length=512, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], generation_config=generation_config)
    answer = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    with open(path, "a", encoding='utf8') as f:
        for j in range(batch_size):
            if i+j < len(data):
                json.dump({i+j: answer[j]}, f, ensure_ascii=False)
                f.write('\n')  # Add a newline to separate entries