# -- coding: utf-8 --
# @Author : wuzixun
# @Time : 2023/5/16 11:24 AM
# @File : finetune.py

import os
import datasets
import torch
import argparse
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
from modeling_chatglm import ChatGLMForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, HfArgumentParser


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatglm_path", type=str, default="/home/xiezizhe/wuzixun/LLM/chatglm-6b")
    parser.add_argument("--output_name", type=str, default="output")
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.chatglm_path, load_in_8bit=True, trust_remote_code=True,
                                      device_map='auto')
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    tokenizer = AutoTokenizer.from_pretrained(args.chatglm_path, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        #     target_modules=["query_key_value"],
        target_modules=['dense', 'dense_h_to_4h', 'dense_4h_to_h', 'query_key_value']
    )

    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True

    dataset_path = "/home/xiezizhe/wuzixun/LLM/Chatgpt-Custom/ChatGLM-LoRA/data/ks_h2h_question"

    dataset = datasets.load_from_disk(dataset_path)

    mini_train_dataset = datasets.Dataset.from_dict(dataset[:20])


    def get_masks_and_position_ids(
            seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
    ):
        mask_position = (
                seq_len - 2
        )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
        attention_mask = torch.ones((1, context_length, context_length), device=device)
        attention_mask.tril_()
        attention_mask[..., : mask_position - 1] = 1
        attention_mask = (attention_mask < 0.5).bool()

        if position_encoding_2d:
            seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
            position_ids = torch.arange(context_length, dtype=torch.long, device=device)
            if not gmask:
                position_ids[seq_length:] = mask_position
            block_position_ids = torch.cat(
                (
                    torch.zeros(seq_length, dtype=torch.long, device=device),
                    torch.arange(
                        context_length - seq_length, dtype=torch.long, device=device
                    )
                    + 1,
                )
            )
            position_ids = torch.stack((position_ids, block_position_ids), dim=0)
        else:
            position_ids = torch.arange(context_length, dtype=torch.long, device=device)
            if not gmask:
                position_ids[context_length - 1:] = mask_position
        return attention_mask, position_ids


    def data_collator(features: list) -> dict:
        len_ids = [len(feature["input_ids"]) for feature in features]
        longest = max(len_ids)
        input_ids = []
        attention_mask_list = []
        position_ids_list = []
        labels_list = []
        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            ids = feature["input_ids"]
            seq_len = feature["seq_len"]
            labels = (
                    [-100] * (seq_len - 1)
                    + ids[(seq_len - 1):]
                    + [-100] * (longest - ids_l)
            )
            ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
            _ids = torch.LongTensor(ids)
            attention_mask, position_ids = get_masks_and_position_ids(
                ids, seq_len, longest, _ids.device, gmask=False
            )
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        attention_mask = torch.stack(attention_mask_list)
        position_ids = torch.stack(position_ids_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


    class MymodifiedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_loss = None
            self.eval_count = 0  # 添加一个计数器

        def compute_loss(self, model, inputs, return_outputs=False):
            loss = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"],
                labels=inputs["labels"],
            ).loss
            self.train_loss = loss.item()
            return loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            self.eval_count += 1  # 每次调用evaluate时，计数器加1
            metrics = None
            # 输出 train_loss
            print(f"Step: {self.state.global_step} Training Loss: {self.train_loss}")
            # 输出调用次数
            print(f"Evaluation Count: {self.eval_count}")
            # 编写自定义评估逻辑
            question = '你是谁'
            response, history = model.chat(tokenizer, question, history=[])
            print('问题:', question)
            print('回答:', response)

            return metrics


    training_args = TrainingArguments(
        "output",
        fp16=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,  # batch size 多加1个就爆炸(要17GB显存)
        learning_rate=1e-4,
        max_steps=300,
        logging_strategy="no",
        remove_unused_columns=False,
        seed=0,
        data_seed=0,
        group_by_length=False,
        evaluation_strategy="steps",  # 添加评估策略
        eval_steps=100,  # 设置评估步数
        save_strategy="no"
        # https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/trainer#transformers.TrainingArguments.save_strategy
    )

    trainer = MymodifiedTrainer(
        model=model,
        train_dataset=mini_train_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()


    def save_tunable_parameters(model, path):
        saved_params = {
            k: v.to("cpu")
            for k, v in model.named_parameters()
            if v.requires_grad
        }
        torch.save(saved_params, path)


    save_tunable_parameters(model, os.path.join("output", "adapter_model.bin"))
