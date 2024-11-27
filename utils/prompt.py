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
# ----------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------
LLAMA2_CHAT_TEMPLETE = """<s>[INST] <<SYS>>
As a smart student, answer the given question. Read paragraph and select only one answer between given choices. Let's think step by step.
<</SYS>>

SOLAR_CHAT_TEMPLETE = """지문을 읽고 질문에 대해 단계별로 추론하며 답을 구하세요.<|im_start|>user
지문: 
{}

질문:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:[/INST]
{}</s>"""
SOLAR_RESPONSE_TEMP = "<|im_start|>assistant"
SOLAR_END_TURN = "</s>"
SOLAR_CHAT_TEMPLETE_PLUS = """지문을 읽고 질문에 대해 단계별로 추론하며 답을 구하세요.<|im_start|>user
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
SOLAR_CHAT_TEMPLETE_R = """<|im_start|>system 
지문을 읽고 질문의 답을 구하세요.<|im_end|>
<|im_start|>user
지문: 
{}

힌트: 
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
SOLAR_CHAT_TEMPLETE_PLUS_R="""<|im_start|>system 
지문을 읽고 질문의 답을 구하세요.<|im_end|>
<|im_start|>user
지문: 
{}

힌트: 
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


QWEN_CHAT_TEMPLETE_EN="""<|im_start|>system 
read paragraph and answer the question.<|im_end|>
<|im_start|>user
pargraph: 
{}

question: 
{}

choices: 
{}

choose one among 1, 2, 3, 4, 5.
answer:<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""
QWEN_CHAT_TEMPLETE_PLUS_EN=""

QWEN_RESPONSE_TEMP_EN="<|im_start|>assistant"
QWEN_END_TURN_EN="<|im_end|>"