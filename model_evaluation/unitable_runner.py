import os
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from torch import nn, Tensor
import tokenizers as tk
import re

from typing import Tuple, List, Sequence, Optional, Union
from functools import partial

from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
from src.utils import (
    subsequent_mask,
    pred_token_within_range,
    greedy_sampling,
    bbox_str_to_token_list,
    cell_str_to_token_list,
    html_str_to_token_list,
    build_table_from_html_and_cell,
    html_table_template,
)
from src.trainer.utils import VALID_HTML_TOKEN, VALID_BBOX_TOKEN, INVALID_CELL_TOKEN

# To Delete
from matplotlib import pyplot as plt
from matplotlib import patches

os.chdir(
    r"C:\Users\jmgarzonv\Desktop\EAFIT\Tesis\models_to_test\unitable\model_evaluation"
)

# Check all model weights have been downloaded to experiments/unitable_weights
MODEL_FILE_NAME = [
    "unitable_large_structure.pt",
    "unitable_large_bbox.pt",
    "unitable_large_content.pt",
]
MODEL_DIR = Path("../experiments/unitable_weights")
assert all(
    [(MODEL_DIR / name).is_file() for name in MODEL_FILE_NAME]
), f"Please download model weights from HuggingFace: https://huggingface.co/poloclub/UniTable/tree/main"

device = torch.device("cuda:0")


def autoregressive_decode(
    model: EncoderDecoder,
    image: Tensor,
    prefix: Sequence[int],
    max_decode_len: int,
    eos_id: int,
    token_whitelist: Optional[Sequence[int]] = None,
    token_blacklist: Optional[Sequence[int]] = None,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(image)
        context = (
            torch.tensor(prefix, dtype=torch.int32).repeat(image.shape[0], 1).to(device)
        )

    for _ in range(max_decode_len):
        eos_flag = [eos_id in k for k in context]
        if all(eos_flag):
            break

        with torch.no_grad():
            causal_mask = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(
                memory, context, tgt_mask=causal_mask, tgt_padding_mask=None
            )
            logits = model.generator(logits)[:, -1, :]

        logits = pred_token_within_range(
            logits.detach(),
            white_list=token_whitelist,
            black_list=token_blacklist,
        )

        next_probs, next_tokens = greedy_sampling(logits)
        context = torch.cat([context, next_tokens], dim=1)
    return context


def image_to_tensor(image: Image, size: Tuple[int, int]) -> Tensor:
    T = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.86597056, 0.88463002, 0.87491087],
                std=[0.20686628, 0.18201602, 0.18485524],
            ),
        ]
    )
    image_tensor = T(image)
    image_tensor = image_tensor.to(device).unsqueeze(0)

    return image_tensor


def rescale_bbox(
    bbox: Sequence[Sequence[float]], src: Tuple[int, int], tgt: Tuple[int, int]
) -> Sequence[Sequence[float]]:
    assert len(src) == len(tgt) == 2
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    bbox = [[int(round(i * j)) for i, j in zip(entry, ratio)] for entry in bbox]
    return bbox


class UniTable:
    def __init__(self):
        self.load_model_structure()
        self.current_model = None

    def load_model_structure(self):
        # UniTable large model
        self.d_model = 768
        patch_size = 16
        nhead = 12
        self.dropout = 0.2

        self.backbone = ImgLinearBackbone(d_model=self.d_model, patch_size=patch_size)
        self.encoder = Encoder(
            d_model=self.d_model,
            nhead=nhead,
            dropout=self.dropout,
            activation="gelu",
            norm_first=True,
            nlayer=12,
            ff_ratio=4,
        )
        self.decoder = Decoder(
            d_model=self.d_model,
            nhead=nhead,
            dropout=self.dropout,
            activation="gelu",
            norm_first=True,
            nlayer=4,
            ff_ratio=4,
        )

    def load_vocab_and_model(
        self,
        vocab_path: Union[str, Path],
        max_seq_len: int,
        model_weights: Union[str, Path],
    ) -> Tuple[tk.Tokenizer, EncoderDecoder]:
        vocab = tk.Tokenizer.from_file(vocab_path)
        model = EncoderDecoder(
            backbone=self.backbone,
            encoder=self.encoder,
            decoder=self.decoder,
            vocab_size=vocab.get_vocab_size(),
            d_model=self.d_model,
            padding_idx=vocab.token_to_id("<pad>"),
            max_seq_len=max_seq_len,
            dropout=self.dropout,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        model.load_state_dict(torch.load(model_weights, map_location="cpu"))
        model = model.to(device)
        return vocab, model

    def table_structure_extraction(self, image_tensor, image):
        if self.current_model != "table_structure":
            self.current_model = "table_structure"
            # Table structure extraction
            self.vocab, self.model = self.load_vocab_and_model(
                vocab_path="../vocab/vocab_html.json",
                max_seq_len=784,
                model_weights=MODEL_DIR / MODEL_FILE_NAME[0],
            )

        # Inference
        pred_html = autoregressive_decode(
            model=self.model,
            image=image_tensor,
            prefix=[self.vocab.token_to_id("[html]")],
            max_decode_len=512,
            eos_id=self.vocab.token_to_id("<eos>"),
            token_whitelist=[self.vocab.token_to_id(i) for i in VALID_HTML_TOKEN],
            token_blacklist=None,
        )

        # Convert token id to token text
        pred_html = pred_html.detach().cpu().numpy()[0]
        pred_html = self.vocab.decode(pred_html, skip_special_tokens=False)
        pred_html = html_str_to_token_list(pred_html)

        return pred_html

    def cell_bbox_extraction(self, image_tensor, image_size, image):
        self.current_model = "bbox"
        # Table cell bbox detection
        vocab, model = self.load_vocab_and_model(
            vocab_path="../vocab/vocab_bbox.json",
            max_seq_len=1024,
            model_weights=MODEL_DIR / MODEL_FILE_NAME[1],
        )

        # Image transformation
        image_tensor = image_to_tensor(image, size=(448, 448))

        # Inference
        pred_bbox = autoregressive_decode(
            model=model,
            image=image_tensor,
            prefix=[vocab.token_to_id("[bbox]")],
            max_decode_len=1024,
            eos_id=vocab.token_to_id("<eos>"),
            token_whitelist=[vocab.token_to_id(i) for i in VALID_BBOX_TOKEN[:449]],
            token_blacklist=None,
        )

        # Convert token id to token text
        pred_bbox = pred_bbox.detach().cpu().numpy()[0]
        pred_bbox = vocab.decode(pred_bbox, skip_special_tokens=False)
        pred_bbox = bbox_str_to_token_list(pred_bbox)
        pred_bbox = rescale_bbox(pred_bbox, src=(448, 448), tgt=image_size)
        return pred_bbox

    def cell_content_extraction(self, image, pred_bbox):
        self.current_model = "content"
        # Table cell content recognition
        vocab, model = self.load_vocab_and_model(
            vocab_path="../vocab/vocab_cell_6k.json",
            max_seq_len=200,
            model_weights=MODEL_DIR / MODEL_FILE_NAME[2],
        )

        # Cell image cropping and transformation
        image_tensor = [
            image_to_tensor(image.crop(bbox), size=(112, 448)) for bbox in pred_bbox
        ]
        image_tensor = torch.cat(image_tensor, dim=0)

        # Inference
        pred_cell = autoregressive_decode(
            model=model,
            image=image_tensor,
            prefix=[vocab.token_to_id("[cell]")],
            max_decode_len=200,
            eos_id=vocab.token_to_id("<eos>"),
            token_whitelist=None,
            token_blacklist=[vocab.token_to_id(i) for i in INVALID_CELL_TOKEN],
        )

        # Convert token id to token text
        pred_cell = pred_cell.detach().cpu().numpy()
        pred_cell = vocab.decode_batch(pred_cell, skip_special_tokens=False)
        pred_cell = [cell_str_to_token_list(i) for i in pred_cell]
        pred_cell = [re.sub(r"(\d).\s+(\d)", r"\1.\2", i) for i in pred_cell]
        return pred_cell

    def predict(self, image):
        image_tensor = image_to_tensor(image, size=(448, 448))

        pred_html = self.table_structure_extraction(image_tensor, image)
        pred_cell = None

        # Uncomment these lines to enable cell bbox detection and content recognition
        # pred_bbox = self.cell_bbox_extraction(image_tensor, image.size, image)
        # pred_cell = self.cell_content_extraction(image, pred_bbox)

        pred_code = build_table_from_html_and_cell(pred_html, pred_cell)
        pred_code = "".join(pred_code)
        pred_code = html_table_template(pred_code)
        return pred_code
