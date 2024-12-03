BASELINE_CHAT_TEMPLETE = """지문을 읽고 질문의 답을 구하세요.<start_of_turn>user
지문:
{}

질문:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""
BASELINE_RESPONSE_TEMP = "<start_of_turn>model"
BASELINE_END_TURN = "<end_of_turn>"
BASELINE_CHAT_TEMPLETE_PLUS = """지문을 읽고 질문의 답을 구하세요.<start_of_turn>user
지문:
{}

질문:
{}

<보기>:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""


QWEN_CHAT_TEMPLETE = """<|im_start|>system 
지문을 읽고 질문의 답을 구하세요.<|im_end|>
<|im_start|>user
지문: 
{}

질문: 
{}

선택지: 
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""
QWEN_CHAT_TEMPLETE_EXP = """<|im_start|>system 
지문을 읽고 단계별로 생각하며 정답을 고르세요.<|im_end|>
<|im_start|>user
지문: 
{}

질문:
{}

선택지: 
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""
QWEN_RESPONSE_TEMP = "<|im_start|>assistant"
QWEN_END_TURN = "<|im_end|>"
QWEN_CHAT_TEMPLETE_PLUS = """<|im_start|>system 
지문을 읽고 질문의 답을 구하세요.<|im_end|>
<|im_start|>user
지문: 
{}

질문: 
{}

<보기>: 
{}

선택지: 
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""


SOLAR_CHAT_TEMPLETE = """As a smart student, answer the given question in Korean. Read paragraph and information, select only one answer between given choices. Let's think step by step.<|im_start|>user
지문: 
{}

질문: 
{}

선택지: 
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""
SOLAR_RESPONSE_TEMP = "<|im_start|>assistant"
SOLAR_END_TURN = "<|im_end|>"
SOLAR_CHAT_TEMPLETE_PLUS = """As a smart student, answer the given question. Read paragraph and select only one answer between given choices. Let's think step by step.<|im_start|>user
지문: 
{}

질문: 
{}

<보기>: 
{}

선택지: 
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""


LLAMA2_CHAT_TEMPLETE = """<s>[INST] <<SYS>>
As a smart student, answer the given question. Read paragraph and select only one answer between given choices. Let's think step by step.
<</SYS>>

지문:
{}

질문:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:[/INST]
{}</s>"""
LLAMA2_RESPONSE_TEMP = "[/INST]"
LLAMA2_END_TURN = "<\s>"
LLAMA2_CHAT_TEMPLETE_PLUS = """<s>[INST] <<SYS>>
As a smart student, answer the given question. Read paragraph and select only one answer between given choices. Let's think step by step.
<</SYS>>

지문:
{}

질문:
{}

<보기>:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:[/INST]
{}</s>"""


LLAMA3_CHAT_TEMPLETE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
As a smart student, answer the given question. Read paragraph and select only one answer between given choices. Let's think step by step.<|eot_id|><|start_header_id|>user<|end_header_id|>
지문:
{}

질문:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|eot_id|>
<|start_header_id|>assistant
{}<|end_header_id|>"""

LLAMA3_RESPONSE_TEMP = "<|start_header_id|>assistant"
LLAMA3_END_TURN = "<|end_header_id|>"
LLAMA3_CHAT_TEMPLETE_PLUS = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
As a smart student, answer the given question. Read paragraph and select only one answer between given choices. Let's think step by step.<|eot_id|><|start_header_id|>user<|end_header_id|>
지문:
{}

질문:
{}

<보기>:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|eot_id|>
<|start_header_id|>assistant
{}<|end_header_id|>"""


EXAONE_CHAT_TEMPLETE = """[|system|] 지문을 읽고 질문의 답을 구하세요.[|endofturn|]
[|user|] 
지문: 
{}

질문 : 
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:[|endofturn|]
[|assistant|]
{}[|endofturn|]"""
EXAONE_RESPONSE_TEMP = "[|assistant|]"
EXAONE_END_TURN = "[|endofturn|]"
EXAONE_CHAT_TEMPLETE_PLUS = """[|system|] 지문을 읽고 질문의 답을 구하세요.[|endofturn|]
[|user|] 
지문: 
{}

질문 : 
{}

<보기>: 
{}

선택지: 
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:[|endofturn|]
[|assistant|]
{}[|endofturn|]"""
