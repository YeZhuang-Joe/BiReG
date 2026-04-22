import os
import re
import json
import requests


# =========================
# 0. 配置路径
# =========================
CONFIG_PATH = "config/api_config.json"
TEMPLATE_EN_PATH = "template/template_en.txt"
TEMPLATE_ZH_PATH = "template/template_zh.txt"


# =========================
# 1. 读取 API Key
# =========================
def load_api_key(config_path: str = CONFIG_PATH, provider: str = "openai") -> str:
    """
    Read API key from config/api_config.json.

    Example:
    {
      "openai_api_key": "your_key_here",
      "deepseek_api_key": "your_key_here"
    }
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Please create it based on api_config_example.json."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if provider == "openai":
        key_name = "openai_api_key"
    elif provider == "deepseek":
        key_name = "deepseek_api_key"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    api_key = config.get(key_name, "").strip()
    if not api_key:
        raise ValueError(f"Missing '{key_name}' in {config_path}")

    return api_key


# =========================
# 2. 语言检测
# =========================
def detect_language(prompt: str) -> str:
    """
    Very simple heuristic:
    if Chinese characters appear, return 'zh', else return 'en'
    """
    for ch in prompt:
        if '\u4e00' <= ch <= '\u9fff':
            return "zh"
    return "en"


# =========================
# 3. 模板路径选择
# =========================
def get_template_path(lang: str) -> str:
    if lang == "zh":
        return TEMPLATE_ZH_PATH
    elif lang == "en":
        return TEMPLATE_EN_PATH
    else:
        raise ValueError(f"Unsupported language: {lang}")


# =========================
# 4. 构建 planner 输入
# =========================
def build_planner_prompt(prompt: str, lang: str) -> str:
    template_path = get_template_path(lang)

    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"Template file not found: {template_path}"
        )

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read().strip()

    return f"{template}\nCaption: {prompt}\nLet's think step by step:"


# =========================
# 5. 解析 LLM 输出
# =========================
def parse_para_dict(output_text: str) -> dict:
    """
    Parse planner output into:
    {
        "Final split ratio": "...",
        "Regional Prompt": "..."
    }

    Supports:
    - English / Chinese labels
    - full-width colon / half-width colon
    - markdown bold markers
    - value on the same line or next line
    """

    # 1) 先去掉 markdown 粗体符号
    cleaned = output_text.replace("**", "").replace("`", "").strip()

    # 2) split ratio：兼容
    #    Final split ratio:
    #    最终分割比例：
    #    并允许值在下一行
    split_ratio_match = re.search(
        r"(?:Final split ratio|最终分割比例)\s*[:：]\s*\n*\s*([\d.,; ]+)",
        cleaned,
        re.IGNORECASE
    )

    # 3) regional prompt：兼容
    #    Regional Prompt:
    #    区域提示词：
    #    区域提示：
    #    并抓取后面所有内容
    prompt_match = re.search(
        r"(?:Regional Prompt|区域提示词|区域提示)\s*[:：]\s*\n*(.*)",
        cleaned,
        re.DOTALL | re.IGNORECASE
    )

    if not split_ratio_match:
        raise ValueError(
            "Failed to parse split ratio. "
            "Expected labels like 'Final split ratio:' or '最终分割比例：'."
        )

    if not prompt_match:
        raise ValueError(
            "Failed to parse regional prompt. "
            "Expected labels like 'Regional Prompt:' / '区域提示词：' / '区域提示：'."
        )

    final_split_ratio = split_ratio_match.group(1).strip()
    regional_prompt = prompt_match.group(1).strip()

    # 4) 清理区域标题，例如：
    # Region0 (Row0, width=0.35):
    regional_prompt = re.sub(
        r"Region\d+\s*\(.*?\)\s*:\s*",
        "",
        regional_prompt,
        flags=re.IGNORECASE
    )

    # 5) 统一 BREAK 格式
    regional_prompt = regional_prompt.replace("\nBREAK\n", " BREAK ")
    regional_prompt = regional_prompt.replace("\nBREAK", " BREAK ")
    regional_prompt = regional_prompt.replace("BREAK\n", " BREAK ")

    # 6) 去掉多余空行和多余空格
    regional_prompt = re.sub(r"\n+", "\n", regional_prompt)
    regional_prompt = re.sub(r"[ \t]+", " ", regional_prompt).strip()

    para_dict = {
        "Final split ratio": final_split_ratio,
        "Regional Prompt": regional_prompt,
    }

    return para_dict


# =========================
# 6. 校验 para_dict
# =========================
def validate_para_dict(para_dict: dict) -> None:
    if not isinstance(para_dict, dict):
        raise TypeError("planner output must be a dict")

    if "Final split ratio" not in para_dict:
        raise KeyError("Missing key: 'Final split ratio'")

    if "Regional Prompt" not in para_dict:
        raise KeyError("Missing key: 'Regional Prompt'")

    split_ratio = para_dict["Final split ratio"]
    regional_prompt = para_dict["Regional Prompt"]

    if not isinstance(split_ratio, str) or not split_ratio.strip():
        raise ValueError("'Final split ratio' must be a non-empty string")

    if not isinstance(regional_prompt, str) or not regional_prompt.strip():
        raise ValueError("'Regional Prompt' must be a non-empty string")

    if "BREAK" not in regional_prompt:
        print("[WARN] 'Regional Prompt' does not contain BREAK. Please confirm planner output format.")


# =========================
# 7. GPT planner
# =========================
def run_gpt_planner(prompt: str, lang: str) -> dict:
    api_key = load_api_key(provider="openai")
    textprompt = build_planner_prompt(prompt, lang)

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": textprompt
            }
        ]
    }
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print("[INFO] Waiting for GPT planner response...")
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    obj = response.json()
    text = obj["choices"][0]["message"]["content"]

    print("\n[RAW LLM OUTPUT]")
    print(text)

    para_dict = parse_para_dict(text)
    validate_para_dict(para_dict)
    return para_dict

def run_deepseek_planner(prompt: str, lang: str) -> dict:
    api_key = load_api_key(provider="deepseek")
    textprompt = build_planner_prompt(prompt, lang)

    url = "https://api.deepseek.com/v1/chat/completions"

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": textprompt
            }
        ]
    }

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print("[INFO] Waiting for DeepSeek planner response...")

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    obj = response.json()
    text = obj["choices"][0]["message"]["content"]

    print("\n[RAW LLM OUTPUT]")
    print(text)

    para_dict = parse_para_dict(text)
    validate_para_dict(para_dict)

    return para_dict
    
# =========================
# 8. Mock planner（最小占位）
# =========================
def run_mock_planner(prompt: str, lang: str) -> dict:
    """
    Minimal mock planner for debugging full_infer.py.
    Later this can be replaced by real LLM output.
    """
    para_dict = {
        "Final split ratio": "0.5,0.5",
        "Regional Prompt": f"{prompt} BREAK {prompt}"
    }
    validate_para_dict(para_dict)
    return para_dict


# =========================
# 9. 统一入口
# =========================
def build_para_dict(prompt: str, lang: str = "auto", planner_name: str = "gpt") -> dict:
    if lang == "auto":
        lang = detect_language(prompt)

    if planner_name == "mock":
        return run_mock_planner(prompt, lang)

    elif planner_name == "gpt":
        return run_gpt_planner(prompt, lang)
    elif planner_name == "deepseek":
        return run_deepseek_planner(prompt, lang)
    else:
        raise ValueError(f"Unsupported planner: {planner_name}")


if __name__ == "__main__":
    test_prompt = "旅人站在左侧，猫在右侧，中间是篝火"
    result = build_para_dict(
        prompt=test_prompt,
        lang="zh",
        planner_name="deepseek"
    )
    print("\n[PARSED PARA_DICT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
