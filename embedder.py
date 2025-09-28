import json
import logging
import os
from functools import partial
from typing import Dict, List, Optional, Union

import math, random
import numpy as np
import queue
import torch
import torch.multiprocessing as mp
from peft import PeftModel
from torch import Tensor, device, nn
from tqdm.autonotebook import tqdm, trange
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer
)
import torch.nn.functional as F
from copy import deepcopy
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from pprint import pprint

logger = logging.getLogger(__name__)


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class General_Embedder(nn.Module):
    def __init__(
        self,
        model: AutoModel,
        normalize=False,
        embed_dim=1792
    ):
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.embed_dim = embed_dim
        self.pool = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        **kwargs,
    ):
        keys = ["normalize", "max_length", "embed_dim"]
        encoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }
        model = SentenceTransformer(model_name_or_path, device=kwargs['device_map'], trust_remote_code=kwargs['trust_remote_code'], model_kwargs={'torch_dtype': kwargs['torch_dtype'],"trust_remote_code": kwargs['trust_remote_code']})
        model.max_seq_length = encoder_args['max_length']
        encoder_args.pop("max_length")
      
        return cls(model=model, **encoder_args)

    def encode(self, 
                sentences: Union[str, List[str]],
                **kwargs):
        
        try:
            target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            ctx = mp.get_context('spawn')
            input_queue = ctx.Queue()
            output_queue = ctx.Queue()
            processes = []

            for cuda_id in target_devices:
                p = ctx.Process(
                    target=self._encode_multi_process_worker,
                    args=(self, cuda_id, input_queue, output_queue),
                    daemon=True
                )
                p.start()
                processes.append(p)

            part_size = math.ceil(len(sentences) / len(processes))
            chunk_size = part_size if part_size < 3200 else 3200

            print(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

            last_chunk_id = 0
            chunk = []

            for sentence in sentences:
                chunk.append(sentence)
                if len(chunk) >= chunk_size:
                    input_queue.put([last_chunk_id, chunk, kwargs])
                    last_chunk_id += 1
                    chunk = []

            if len(chunk) > 0:
                input_queue.put([last_chunk_id, chunk, kwargs])
                last_chunk_id += 1

            results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
            embeddings = np.concatenate([result[1] for result in results_list])
            
            for p in processes:
                p.terminate()
            for p in processes:
                p.join()
                p.close()
            input_queue.close()
            output_queue.close()
            torch.cuda.empty_cache()

            return embeddings
        except RuntimeError as e:
            print(e)
            print('Multi-Process运行错误，使用单进程推理！')
            return self._encode(sentences, **kwargs)
    
    @staticmethod
    def _encode_multi_process_worker(self, target_device, input_queue, results_queue):

        while True:
            try:
                last_chunk_id, sentences, kwargs = input_queue.get()
                kwargs.update(device=target_device)
                embeddings = self._encode(sentences, **kwargs)
                results_queue.put([last_chunk_id, embeddings])
            except queue.Empty:
                break
    
    @torch.no_grad()
    def _encode(
        self,
        sentences: Union[str, List[str]],
        prompt: str = '',
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
    ):
        batch_size = 128

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(device)

        if convert_to_tensor:
            convert_to_numpy = False

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []

        for start_index in trange(
            0,
            len(sentences),
            batch_size,
            desc=f"Batches_{device}",
            disable=not show_progress_bar,
            position=int(device[-1]) if device[-1].isdigit() else 0
        ):
            sentences_batch = sentences_sorted[
                start_index : start_index + batch_size
            ]
            
            sentences_batch = [(x if x != '' else (x + 'Null')) for x in sentences_batch]

            task_prompt = "This sentence: <|im_start|>“{text}” means in one word: “"
            sentences_batch = [task_prompt.format(text=(prompt + x)) for x in sentences_batch]

            embeddings = self.model.encode(sentences_batch, normalize_embeddings=False)
            if self.normalize:
                embeddings = normalize(embeddings[:, :self.embed_dim])
            else:
                embeddings = embeddings[:, :self.embed_dim]
            all_embeddings.append(torch.tensor(embeddings))

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
 
        return len(text)
