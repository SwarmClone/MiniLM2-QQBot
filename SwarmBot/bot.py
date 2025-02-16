import sys
import os
import json
import math
import torch # type: ignore
from tqdm import tqdm
from torch.nn import functional as F # type: ignore
from model import NGPT, RWKV7 # type: ignore
from tokenizers import Tokenizer # type: ignore
import nonebot # type: ignore
from nonebot.typing import T_State # type: ignore
from nonebot.params import EventPlainText # type: ignore
from nonebot.rule import to_me # type: ignore
from nonebot.plugin import on_message, on_command # type: ignore
from nonebot.adapters.onebot.v11 import Adapter as v11Adapter # type: ignore
from nonebot.adapters.console import Adapter as ConsoleAdapter # type: ignore

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(v11Adapter)
driver.register_adapter(ConsoleAdapter)

history: list[tuple[str, str]] = []

clear = on_command("clear", aliases={"cls", "清除"}, block=True)
@clear.handle()
async def handle_clear():
    global history
    history = []

question_command = on_command("MiniLM", aliases={"minilm"}, block=True)
@question_command.handle()
async def handel_command(args = EventPlainText()):
    global history
    history = append_history(history, "human", args.strip())
    input_ids = build_context(history, tokenizer, train_config['max_length'],
                            system_prompt=train_config.get('system_prompt')).to("cuda")
    response = ""
    n_blankline = 0
    with torch.no_grad():
        while True:
            output = model(input_ids)
            logits = F.softmax(output[0][-1] / train_config['temperature'], dim=-1)
            # 采样输出，取概率最高的n个进行加权随机采样
            probs, indices = logits.topk(round(vocab_size * train_config['top_p']))
            sample = torch.multinomial(probs, 1)
            token_id = indices[sample]
            input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=1)[:, -train_config['max_length']:] # 自回归生成
            token = tokenizer.id_to_token(token_id.item())
            if token == "\n":
                n_blankline += 1
                if n_blankline >= 3:
                    break
            else:
                n_blankline = 0
            response += token
    response = response.strip()
    history = append_history(history, "ai", response)
    await question_command.finish(response)

question_at = on_message(rule=to_me(), block=True)
@question_at.handle()
async def handle_at(args = EventPlainText()):
    global history
    history = append_history(history, "human", args.strip())
    input_ids = build_context(history, tokenizer, train_config['max_length'],
                            system_prompt=train_config.get('system_prompt')).to("cuda")
    response = ""
    n_blankline = 0
    with torch.no_grad():
        while True:
            output = model(input_ids)
            logits = F.softmax(output[0][-1] / train_config['temperature'], dim=-1)
            # 采样输出，取概率最高的n个进行加权随机采样
            probs, indices = logits.topk(round(vocab_size * train_config['top_p']))
            sample = torch.multinomial(probs, 1)
            token_id = indices[sample]
            input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=1)[:, -train_config['max_length']:] # 自回归生成
            token = tokenizer.id_to_token(token_id.item())
            if token == "\n":
                n_blankline += 1
                if n_blankline >= 3:
                    break
            else:
                n_blankline = 0
            response += token
    response = response.strip()
    history = append_history(history, "ai", response)
    await question_at.finish(response)

def build_context(history: list[tuple[str, str]], tokenizer: Tokenizer,
                max_length: int, system_prompt: str | None = None) -> torch.Tensor:
    ids = []
    human_prefix_ids = tokenizer.encode("人类：").ids
    ai_prefix_ids = tokenizer.encode("AI：").ids
    separator_ids = tokenizer.encode("\n" * 3).ids
    system_prompt_ids = tokenizer.encode(system_prompt).ids + separator_ids if system_prompt else []
    for i in range(len(history)):
        turn = history[i]
        ids += human_prefix_ids + tokenizer.encode(turn[0]).ids + separator_ids
        ids += ai_prefix_ids + tokenizer.encode(turn[1]).ids
        if i < len(history) - 1:
            ids += separator_ids
    ids = system_prompt_ids + ids[len(system_prompt_ids)-max_length:]
    return torch.LongTensor(ids).unsqueeze(0)

def append_history(history: list[tuple[str, str]], role: str, text: str) -> list[tuple[str, str]]:
    if role == "human":
        history.append((text, ""))
    else:
        history[-1] = (history[-1][0], text)
    return history

if __name__ == "__main__":
    config_path = "../models/dialogue/dialogue.json"
    config_dir = os.path.dirname(config_path) # 配置文件路径
    train_config = json.load(open(config_path))

    # 加载tokenizer并获取词表大小
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(os.path.join(config_dir, train_config['tokenizer_path']))
    vocab_size = tokenizer.get_vocab_size()
    print(f"==> Vocab size: {vocab_size}")

    # 根据配置文件创建模型
    model_type = train_config["model"]
    print(f"Loading {model_type} model...")
    model: torch.nn.Module
    if model_type == "NGPT":
        model = NGPT(
            vocab_size=vocab_size,
            dim=train_config['model_dim'],
            max_length=train_config['max_length'],
            n_heads=train_config['num_heads'],
            n_blocks=train_config['num_layers'],
            dropout=0 # 推理时不使用dropout
        )
    elif model_type == "RWKV7":
        model = RWKV7(
            vocab_size=2 ** math.ceil(math.log2(vocab_size)), # 确保vocab_size为2的幂
            dim=train_config['model_dim'],
            n_blocks=train_config['num_layers'],
            max_lr=train_config['max_learning_rate']
        )
    # 统计参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"==> Number of parameters: {params / 1e6:.2f}M")
    # 加载已有的检查点
    if train_config['checkpoint_file']:
        checkpoint_path = os.path.join(config_dir, train_config['checkpoint_file'])
        print(f"==> Loading checkpoint from {checkpoint_path}, step={train_config['checkpoint_step']}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    # 将模型移动到显存
    model.to("cuda")
    model.eval()

    torch.set_float32_matmul_precision('high') # 调整精度以加速推理

    nonebot.run()
