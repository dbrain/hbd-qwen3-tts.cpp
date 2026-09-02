import json, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ.get("BREEZE_CKPT","/models/Breeze-TTS-2-hf"))
CASES = [
    "Hello world.",
    "[S0]The quick brown fox jumps over the lazy dog.",
    "[S0]<ins_bos>Speak clearly and naturally.<ins_eos>The quick brown fox jumps over the lazy dog.",
    "  leading and   internal   spaces  ",
    "Numbers: 1234567890, 3.14159, -42.",
    "Punctuation!?;:'\"()[]{}<>@#$%^&*",
    "Mixed CASE and CamelCase and snake_case and kebab-case",
    "Unicode: café naïve résumé — em-dash … ellipsis “curly quotes”",
    "中文测试：你好，世界！",
    "Emoji: 🎧🔥👍 and math ∑∫≈",
    "Tabs\tand\nnewlines\n\nand\r\ncarriage",
    "A very long sentence that goes on and on, deliberately, so that the tokenizer has to make a lot of merge decisions across many words and punctuation marks, including some rare ones like antidisestablishmentarianism and pneumonoultramicroscopicsilicovolcanoconiosis.",
    "<|AUDIO|><|AUDIO|><|audio_eos|>",
    "[S3]Hi.<|AUDIO|><|AUDIO|><|audio_eos|>[S3]Bye.",
    "<pad><eos><bos><unk><mask>",
    "<unused0> literal unused token",
    "trailing space ",
    " leading space",
    "",
    "a",
    "  ",
    "The rain in Spain falls mainly on the plain; it's a lovely day, isn't it? (Yes.)",
    "Dr. Smith went to Washington D.C. on Jan. 3rd, 2026 at 5:30 p.m.",
    "one▁two",   # literal metaspace already present
]
out = []
for c in CASES:
    out.append({"text": c,
                "ids_special": tok(c, add_special_tokens=True)["input_ids"],
                "ids_plain":   tok(c, add_special_tokens=False)["input_ids"]})
print(json.dumps(out, ensure_ascii=False, indent=1))
