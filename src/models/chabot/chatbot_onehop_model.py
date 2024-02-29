"""
@Desc:
@Reference:
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler

from transformers.models.bart import modeling_bart
from transformers.models.bart.modeling_bart import BartConfig
from transformers import LlamaTokenizer,LlamaConfig

from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.utils import nlg_eval_utils
from src.modules.chatbot.chatbot_onehop import OneHopGNNBartForCG
from src.utils.charbot import model_utils
from src.models.lightning_base import BaseTransformer
from src.models.chabot.bart import MyBart
from src.modules.chatbot.dataloader_onehop import (
    CommonGraphDataset
)
from peft import PeftModel, PeftConfig
from peft import get_peft_model, LoraConfig, TaskType
logger = logging.getLogger(__name__)


class OneHopGNNBart(MyBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: LlamaConfig = LlamaConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.bos_token_id = self.tokenizer.bos_token_id
        # self.tokenizer.model_max_length = 512
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, OneHopGNNBartForCG, self.config)
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.1)
        self.model = get_peft_model(self.model, self.peft_config) #lora disabled
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = CommonGraphDataset
        self.training_type = self.hparams.training_type
        print('freeze parameters')
        # for name, param in self.model.model.named_parameters():
        #     if name != 'layers': 
        #         param.requires_grad = False
        # for name, param in self.model.model.named_parameters():
        #     if name != 'layers': 
        #         param.requires_grad = False

    def forward(self, **kwargs):
        return self.model( **kwargs)
    #has some problem here  
    def _step(self, batch: dict):
        src_ids = batch['src_ids']
        tgt_ids = batch['tgt_ids']
        encoder_attention = batch['encoder_attention']
        decoder_attention = batch['decoder_attention']
        one_hop_graph = batch['graph']

        outputs = self(one_hop_graph=one_hop_graph,
                       src_ids=src_ids,
                       tgt_ids=tgt_ids,
                       encoder_attention=encoder_attention,
                       decoder_attention=decoder_attention)
        
        loss = outputs["loss"]
        #breakpoint()
        return loss

    @torch.no_grad()
    def _generative_step(self, batch: dict) -> dict:
        tik = datetime.now()
        '''src_ids, src_mask = batch['input_ids'], batch["attention_mask"]
        item_nums = batch['item_nums']
        encoder_outputs = self.model.get_encoder()(input_ids=src_ids, attention_mask=src_mask, item_nums=item_nums)'''
        src_ids = batch['src_ids']
        tgt_ids = batch['tgt_ids']
        encoder_attention = batch['encoder_attention']
        decoder_attention = batch['decoder_attention']
        one_hop_graph = batch['graph']
        csk_ids = []
        for _one_hop_graph in one_hop_graph:
            _csk_ids = _one_hop_graph.ndata['h'][:, [0, 3]]
            _csk_ids = set([int(ids) for tri in _csk_ids for ids in tri])
            csk_ids.append(_csk_ids)

        # transformers.generation_utils
        extra_params = {}
        if self.hparams.num_beam_groups > 1:
            extra_params["num_beam_groups"] = self.hparams.num_beam_groups
            extra_params["diversity_penalty"] = self.hparams.diversity_penalty
        if self.eval_beams >= 1:
            extra_params["num_beams"] = self.eval_beams
        if self.hparams.repetition_penalty > 0:
            extra_params["repetition_penalty"] = self.hparams.repetition_penalty

        generated_ids = self.model.generate(
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
        

        tok = datetime.now()
        batch_gen_time = tok - tik
        preds: List[str] = self.gen_ids_to_clean_text(generated_ids)
        targets: List[str] = self.gen_ids_to_clean_text(tgt_ids)
        loss = self._step(batch)

        base_metrics = {"loss": loss.item()}

        rouge_metrics: Dict = nlg_eval_utils.calculate_rouge(pred_lines=preds, tgt_lines=targets)
        base_metrics.update(**rouge_metrics)
        bleu_metrics: Dict = nlg_eval_utils.calculate_bleu(ref_lines=[self.tokenizer.tokenize(l) for l in targets],
                                                           gen_lines=[self.tokenizer.tokenize(l) for l in preds])
        base_metrics.update(**bleu_metrics)

        entity_score_metrics: Dict = nlg_eval_utils.compute_ent_score(src_ids, generated_ids, csk_ids)

        base_metrics.update(**entity_score_metrics)
        summ_len = np.mean(list(map(len, generated_ids)))       
        # update metric_names
        self.update_metric_names(base_metrics, update_flag=self.metric_names_update_flag)
        self.metric_names_update_flag = False
        base_metrics.update(batch_gen_time=batch_gen_time, gen_len=summ_len,
                            preds=preds, targets=targets)
        return base_metrics

    def training_step(self, batch, batch_idx) -> Dict:
        outputs = self._step(batch)
        loss = outputs
        logs = {"loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        logs["batch_size"] = batch['src_ids'].shape[0]
        #breakpoint()
        return {"loss": loss, "log": logs}

    def get_dataset(self, data_name: str, batch_size: int) -> CommonGraphDataset:
        dataset = self.dataset_class(
            data_dir=self.hparams.data_dir,
            hparams=self.hparams,
            batch_size=batch_size,
            tokenizor=self.tokenizer,
            embedding_size=self.config.hidden_size,
            data_name=data_name,
        )

        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

    def on_after_backward(self):
        # This method is called after the backward pass
        for name, param in self.named_parameters():
            if param.grad is not None and 'embed' in name: #(np.isnan(param.grad.norm().item())):
                print('!'*10)
                print(f"{name}: {param.grad.norm().item()}")
                print('!'*10)
        breakpoint()
