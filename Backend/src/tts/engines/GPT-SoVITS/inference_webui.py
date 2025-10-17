import logging
import traceback
import warnings
import torchaudio

# ... (all initial logging setup and warnings) ...
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)

import json
import os
import re
import sys
import torch
from text.LangSegmenter import LangSegmenter

# <<< START: CUDA OPTIMIZATIONS >>>
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("[INFO] CUDA optimizations are enabled.")
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        print("[INFO] FlashAttention/SDP kernels enabled for faster Transformer operations.")
    except Exception as e:
        print(f"[WARN] Could not enable SDP kernels: {e}")
else:
    device = "cpu"
# <<< END: CUDA OPTIMIZATIONS >>>

# ... (all the path and config setup code) ...
current_script_directory = os.path.dirname(os.path.abspath(__file__))
pretrained_models_dir = os.path.join(current_script_directory, "pretrained_models")

version = model_version = os.environ.get("version", "v2")
path_sovits_v3 = os.path.join(pretrained_models_dir, "s2Gv3.pth")
path_sovits_v4 = os.path.join(pretrained_models_dir, "gsv-v4-pretrained/s2Gv4.pth")
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)
pretrained_sovits_name = [
    os.path.join(pretrained_models_dir, "s2G488k.pth"),
    os.path.join(pretrained_models_dir, "gsv-v2final-pretrained/s2G2333k.pth"),
    os.path.join(pretrained_models_dir, "s2Gv3.pth"),
    os.path.join(pretrained_models_dir, "gsv-v4-pretrained/s2Gv4.pth"),
]
pretrained_gpt_name = [
    os.path.join(pretrained_models_dir, "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"),
    os.path.join(pretrained_models_dir, "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
    os.path.join(pretrained_models_dir, "s1v3.ckpt"),
    os.path.join(pretrained_models_dir, "s1v3.ckpt"),
]

_ = [[], []]
for i in range(4):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
pretrained_gpt_name, pretrained_sovits_name = _

if not os.path.exists("./weight.json"):
    with open("./weight.json", "w", encoding="utf-8") as file:
        json.dump({"GPT": {}, "SoVITS": {}}, file)

with open("./weight.json", "r", encoding="utf-8") as file:
    weight_data = json.load(file)
    gpt_path = os.environ.get("gpt_path", weight_data.get("GPT", {}).get(version, pretrained_gpt_name))
    sovits_path = os.environ.get("sovits_path", weight_data.get("SoVITS", {}).get(version, pretrained_sovits_name))
    if isinstance(gpt_path, list):
        gpt_path = gpt_path[0] if gpt_path else None
    if isinstance(sovits_path, list):
        sovits_path = sovits_path[0] if sovits_path else None

cnhubert_base_path = os.environ.get("cnhubert_base_path", os.path.join(pretrained_models_dir, "chinese-hubert-base"))
bert_path = os.environ.get("bert_path", os.path.join(pretrained_models_dir, "chinese-roberta-wwm-ext-large"))

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

import librosa
import numpy as np
from feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
cnhubert.cnhubert_base_path = cnhubert_base_path
import random
from module.models import SynthesizerTrn, SynthesizerTrnV3, Generator

def set_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000000)
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

from time import time as ttime
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from peft import LoraConfig, get_peft_model
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

dict_language_v1 = {"中文": "all_zh", "英文": "en", "日文": "all_ja", "中英混合": "zh", "日英混合": "ja", "多语种混合": "auto"}
dict_language_v2 = {"中文": "all_zh", "英文": "en", "日文": "all_ja", "粤语": "all_yue", "韩文": "all_ko", "中英混合": "zh", "日英混合": "ja", "粤英混合": "yue", "韩英混合": "ko", "多语种混合": "auto", "多语种混合(粤语)": "auto_yue"}
dict_language = dict_language_v1 if version == "v1" else {i18n(key): value for key, value in dict_language_v2.items()}


tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0][1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = [res[i].repeat(word2ph[i], 1) for i in range(len(word2ph))]
    return torch.cat(phone_level_feature, dim=0).T

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

resample_transform_dict = {}
def resample(audio_tensor, sr0, sr1):
    key = f"{sr0}-{sr1}"
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
v3v4set = {"v3", "v4"}

def change_sovits_weights(sovits_path):
    global vq_model, hps, version, model_version, dict_language, if_lora_v3
    if not sovits_path or not os.path.exists(sovits_path):
        print(f"SoVITS path is invalid or not provided: {sovits_path}. Skipping model change.")
        return

    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    print(f"Loading SoVITS model: {sovits_path}, Version: {version}, Model Version: {model_version}, LoRA: {if_lora_v3}")
    
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    if if_lora_v3 and not is_exist:
        raise FileExistsError(f"SoVITS {model_version} base model is missing for LoRA.")
        
    dict_language = dict_language_v1 if version == "v1" else {i18n(key): value for key, value in dict_language_v2.items()}
    
    dict_s2 = load_sovits_new(sovits_path)
    hps = DictToAttrRecursive(dict_s2["config"])
    hps.model.semantic_frame_rate = "25hz"
    version = hps.model.version = "v2" if "enc_p.text_embedding.weight" not in dict_s2["weight"] else ("v1" if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322 else "v2")
    
    if model_version not in v3v4set:
        vq_model = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model)
        model_version = version
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model)
    
    if "pretrained" not in sovits_path:
        try: del vq_model.enc_q
        except: pass
            
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    
    if not if_lora_v3:
        print(f"Loading SoVITS_{model_version}", vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else:
        base_model_path = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        print(f"Loading SoVITS_{model_version} pretrained G", vq_model.load_state_dict(load_sovits_new(base_model_path)["weight"], strict=False))
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(target_modules=["to_k", "to_q", "to_v", "to_out.0"], r=lora_rank, lora_alpha=lora_rank, init_lora_weights=True)
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print(f"Loading SoVITS_{model_version}_lora{lora_rank}")
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        
    with open("./weight.json", "r") as f: data = json.load(f)
    data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as f: json.dump(data, f)

def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    if not gpt_path or not os.path.exists(gpt_path):
        print(f"GPT path is invalid or not provided: {gpt_path}. Skipping model change.")
        return
        
    print(f"Loading GPT model: {gpt_path}")
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()

    with open("./weight.json", "r") as f: data = json.load(f)
    data["GPT"][version] = gpt_path
    with open("./weight.json", "w") as f: json.dump(data, f)

def init_bigvgan():
    global bigvgan_model, hifigan_model
    from BigVGAN import bigvgan
    print("Initializing BigVGAN vocoder...")
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(os.path.join(pretrained_models_dir, "models--nvidia--bigvgan_v2_24khz_100band_256x"), use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    if hifigan_model:
        hifigan_model = hifigan_model.cpu(); hifigan_model = None; torch.cuda.empty_cache()
    if is_half:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)

def init_hifigan():
    global hifigan_model, bigvgan_model
    print("Initializing HiFiGAN vocoder...")
    hifigan_model = Generator(initial_channel=100, resblock="1", resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], upsample_rates=[10, 6, 2, 2, 2], upsample_initial_channel=512, upsample_kernel_sizes=[20, 12, 4, 4, 4], gin_channels=0, is_bias=True)
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load(os.path.join(pretrained_models_dir, "gsv-v4-pretrained/vocoder.pth"), map_location="cpu")
    print("Loading vocoder weights:", hifigan_model.load_state_dict(state_dict_g))
    if bigvgan_model:
        bigvgan_model = bigvgan_model.cpu(); bigvgan_model = None; torch.cuda.empty_cache()
    if is_half:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)

from module.mel_processing import mel_spectrogram_torch, spectrogram_torch
spec_min, spec_max = -12, 2
def norm_spec(x): return (x - spec_min) / (spec_max - spec_min) * 2 - 1
def denorm_spec(x): return (x + 1) / 2 * (spec_max - spec_min) + spec_min
mel_fn = lambda x: mel_spectrogram_torch(x, n_fft=1024, num_mels=100, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=None, center=False)
mel_fn_v4 = lambda x: mel_spectrogram_torch(x, n_fft=1280, num_mels=100, sampling_rate=32000, hop_size=320, win_size=1280, fmin=0, fmax=None, center=False)

def get_spepc(hps, filename):
    audio, sr = librosa.load(filename, sr=int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    if audio.abs().max() > 1:
        audio /= audio.abs().max()
    spec = spectrogram_torch(audio.unsqueeze(0), hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
    return spec

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language.replace("all_", ""), version)
    return cleaned_text_to_sequence(phones, version), word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)
    return bert

dtype = torch.float16 if is_half else torch.float32
splits = {"。", "！", "？", "…", ",", ".", "!", "?", "~", ":", "：", "—"}
from text import chinese

def get_phones_and_bert(text, language, version, final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        formattext = text.replace("  ", " ")
        if language == "all_zh" and re.search(r"[A-Za-z]", formattext):
            formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(formattext, "zh", version)
        elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
            formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(formattext, "yue", version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros((1024, len(phones)), dtype=dtype).to(device)
            if language == "all_zh":
                bert = get_bert_feature(norm_text, word2ph).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist, langlist = [], []
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                lang = "yue" if tmp["lang"] == "zh" else tmp["lang"]
                langlist.append(lang)
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(language if tmp["lang"] != "en" else "en")
                textlist.append(tmp["text"])
        
        phones_list, bert_list, norm_text_list = [], [], []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)
        
    if not final and len(phones) < 6:
        return get_phones_and_bert(f".{text}", language, version, final=True)

    return phones, bert.to(dtype), norm_text

def merge_short_text_in_array(texts, threshold):
    if len(texts) < 2: return texts
    result, text = [], ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if text:
        if not result:
            result.append(text)
        else:
            result[-1] += text
    return result

cache = {}
ref_audio_cache = {}

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("按标点符号切"), top_k=50, top_p=1, temperature=1, ref_free=False, speed=1, if_freeze=False, inp_refs=None, sample_steps=16, pause_second=0.2):
    from queue import Queue
    from threading import Thread

    global cache, ref_audio_cache
    if not ref_wav_path: raise ValueError("Reference audio path is missing.")
    if not text: raise ValueError("TTS text is missing.")
    
    if prompt_text is None or len(prompt_text) == 0: ref_free = True
    if model_version in v3v4set: ref_free = False
    
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text and prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    
    text = text.strip("\n")
    print(i18n("实际输入的目标文本:"), text)
    
    zero_wav_torch = torch.zeros(int(hps.data.sampling_rate * pause_second), dtype=dtype, device=device)
        
    if not ref_free:
        if ref_wav_path in ref_audio_cache:
            print(f"[INFO] Reference audio cache hit for {ref_wav_path}")
            prompt, refer_spec, ref_audio_tensor, sr_ref = ref_audio_cache[ref_wav_path]
        else:
            print(f"[INFO] Reference audio cache miss for {ref_wav_path}. Processing...")
            ref_audio_tensor, sr_ref = torchaudio.load(ref_wav_path)
            ref_audio_tensor = ref_audio_tensor.to(device).float()
            if ref_audio_tensor.shape[0] > 1:
                ref_audio_tensor = ref_audio_tensor.mean(0, keepdim=True)

            with torch.inference_mode():
                wav16k = resample(ref_audio_tensor, sr_ref, 16000)
                if len(wav16k[0]) > 160000 or len(wav16k[0]) < 48000:
                    raise ValueError(i18n("参考音频在3~10秒范围外，请更换！"))
                wav16k = wav16k.half() if is_half else wav16k
                
                zero_wav_16k = torch.zeros(int(16000 * pause_second), dtype=wav16k.dtype, device=device)
                ssl_input = torch.cat([wav16k, zero_wav_16k.unsqueeze(0)], dim=1)
                ssl_content = ssl_model.model(ssl_input)["last_hidden_state"].transpose(1, 2)
                codes = vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to(device)

            wav_hps_sr = resample(ref_audio_tensor, sr_ref, hps.data.sampling_rate)
            audio_hps_sr_norm = wav_hps_sr
            if audio_hps_sr_norm.abs().max() > 1:
                audio_hps_sr_norm /= audio_hps_sr_norm.abs().max()
            spec = spectrogram_torch(audio_hps_sr_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
            refer_spec = spec.to(dtype).to(device)

            ref_audio_cache[ref_wav_path] = (prompt, refer_spec, ref_audio_tensor, sr_ref)

    if how_to_cut == i18n("凑四句一切"): text = cut1(text)
    elif how_to_cut == i18n("凑15字一切"): text = cut_by_fixed_length(text)
    elif how_to_cut == i18n("按中文句号。切"): text = cut3(text)
    elif how_to_cut == i18n("按英文句号.切"): text = cut4(text)
    elif how_to_cut == i18n("按标点符号切"): text = cut5(text)
    text = text.replace("\n\n", "\n")
    texts = text.split("\n")
    texts = [t for t in texts if t.strip()]
    texts = merge_short_text_in_array(texts, 5)

    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)

    audio_queue = Queue()

    def producer():
        nonlocal ref_audio_tensor
        try:
            for i_text, current_segment_text in enumerate(texts):
                if not current_segment_text.strip(): continue
                if current_segment_text[-1] not in splits:
                    current_segment_text += "。" if text_language != "en" else "."
                
                phones2, bert2, norm_text2 = get_phones_and_bert(current_segment_text, text_language, version)
                
                if not ref_free:
                    bert = torch.cat([bert1, bert2], 1)
                    all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
                else:
                    bert = bert2
                    all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

                if i_text in cache and if_freeze:
                    pred_semantic = cache[i_text]
                else:
                    with torch.inference_mode():
                        pred_semantic, idx = t2s_model.model.infer_panel(all_phoneme_ids, all_phoneme_len, prompt if not ref_free else None, bert, top_k=top_k, top_p=top_p, temperature=temperature, early_stop_num=hz * max_sec)
                    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                    if if_freeze: cache[i_text] = pred_semantic
                
                with torch.inference_mode():
                    audio_segment = None
                    if model_version not in v3v4set:
                        refers = [refer_spec] if ref_wav_path else []
                        if refers:
                            audio_segment = vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed)[0][0]
                    else:
                        refer = refer_spec
                        phoneme_ids0 = torch.LongTensor(phones1 if not ref_free else []).to(device).unsqueeze(0)
                        phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
                        
                        fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0) if not ref_free else None, phoneme_ids0, refer)
                        
                        tgt_sr_vocoder = 24000 if model_version == "v3" else 32000
                        if sr_ref != tgt_sr_vocoder:
                            ref_audio_tensor = resample(ref_audio_tensor, sr_ref, tgt_sr_vocoder)
                        
                        mel2 = (mel_fn if model_version == "v3" else mel_fn_v4)(ref_audio_tensor)
                        mel2 = norm_spec(mel2)
                        T_min = min(mel2.shape[2], fea_ref.shape[2])
                        mel2 = mel2[:, :, :T_min]
                        fea_ref = fea_ref[:, :, :T_min]
                        Tref = 468 if model_version == "v3" else 500
                        Tchunk = 934 if model_version == "v3" else 1000
                        if T_min > Tref:
                            mel2, fea_ref = mel2[:, :, -Tref:], fea_ref[:, :, -Tref:]
                            T_min = Tref
                        
                        mel2 = mel2.to(dtype)
                        fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
                        
                        cfm_resss = []
                        idx_cfm = 0
                        while True:
                            chunk_len = Tchunk - T_min
                            fea_todo_chunk = fea_todo[:, :, idx_cfm : idx_cfm + chunk_len]
                            if fea_todo_chunk.shape[-1] == 0: break
                            idx_cfm += fea_todo_chunk.shape[-1]
                            
                            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                            cfm_res = vq_model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0)
                            cfm_res = cfm_res[:, :, mel2.shape[2]:]
                            mel2 = cfm_res[:, :, -T_min:]
                            fea_ref = fea_todo_chunk[:, :, -T_min:]
                            cfm_resss.append(cfm_res)

                        if cfm_resss:
                            cfm_res_final = torch.cat(cfm_resss, 2)
                            cfm_res_final = denorm_spec(cfm_res_final)
                            vocoder_model = bigvgan_model if model_version == "v3" else hifigan_model
                            if not vocoder_model:
                                init_bigvgan() if model_version == "v3" else init_hifigan()
                                vocoder_model = bigvgan_model if model_version == "v3" else hifigan_model
                            audio_segment = vocoder_model(cfm_res_final)[0][0]
                
                if audio_segment is not None and audio_segment.numel() > 0:
                    audio_segment = audio_segment.clamp(-1, 1)
                    audio_queue.put(((24000 if model_version == "v3" else 32000), (audio_segment.cpu().numpy() * 32767).astype(np.int16)))

                    if pause_second > 0:
                        audio_queue.put((hps.data.sampling_rate, (zero_wav_torch.cpu().numpy() * 32767).astype(np.int16)))
        finally:
            audio_queue.put(None)

    producer_thread = Thread(target=producer)
    producer_thread.start()

    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        yield chunk
    
    producer_thread.join()

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text and todo_text[-1] not in splits: todo_text += "。"
    i_split_head = i_split_tail = 0
    todo_texts = []
    while i_split_head < len(todo_text):
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def cut1(inp):
    inps = split(inp.strip("\n"))
    split_idx = list(range(0, len(inps), 4))
    split_idx.append(len(inps))
    return "\n".join(["".join(inps[split_idx[i]:split_idx[i+1]]) for i in range(len(split_idx)-1) if inps[split_idx[i]:split_idx[i+1]]])

def cut2(inp):
    inps = split(inp.strip("\n"))
    if len(inps) < 2: return inp
    opts, summ, tmp_str = [], 0, ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 2:
            opts.append(tmp_str)
            summ, tmp_str = 0, ""
    if tmp_str: opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 2:
        opts[-2] += opts[-1]
        opts.pop()
    return "\n".join(opts)

def cut3(inp): return "\n".join(inp.strip("\n").strip("。").split("。"))
def cut4(inp): return "\n".join(re.split(r"(?<!\d)\.(?!\d)", inp.strip("\n").strip(".")))

def cut_by_fixed_length(inp, max_len=50):
    """Cuts text into chunks of a maximum length, trying to split at spaces."""
    inp = inp.strip()
    texts = []
    while len(inp) > max_len:
        # Find the last space within the limit
        cut_pos = inp.rfind(' ', 0, max_len)
        # If no space is found, we have to break the word
        if cut_pos <= 0:
            cut_pos = max_len
        # Get the chunk and the rest of the text
        chunk = inp[:cut_pos]
        inp = inp[cut_pos:].lstrip()
        texts.append(chunk)
    # Add the last remaining part
    if inp:
        texts.append(inp)
    return "\n".join(texts)

def cut5(inp):
    inp = inp.strip("\n")
    end_punc = {"。", "！", "？", "!", "?"}
    mid_punc = {",", "，", "：", ":", ";", "；", "…"}
    sentences = []
    current_sentence = ""
    for char in inp:
        current_sentence += char
        if char in end_punc or char in mid_punc:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    return "\n".join(filter(None, sentences))

# --- Automatic Model Initialization on Import ---
print("--- Initializing Core Models ---")
try:
    if sovits_path and os.path.exists(sovits_path):
        change_sovits_weights(sovits_path)
    else:
        print(f"[WARN] SoVITS path not found or not set: {sovits_path}.")
    if gpt_path and os.path.exists(gpt_path):
        change_gpt_weights(gpt_path)
    else:
        print(f"[WARN] GPT path not found or not set: {gpt_path}.")
    
    bigvgan_model = hifigan_model = None
    if 'model_version' in globals():
        if model_version == "v3": init_bigvgan()
        if model_version == "v4": init_hifigan()
    print("--- Core Models Initialized ---")
except Exception:
    print("[ERROR] A critical error occurred during initial model loading.")
    traceback.print_exc()