BASELINE_CHAT_TEMPLETE = """지문을 읽고 질문에 대해 단계별로 추론하여 답을 구하세요.<start_of_turn>user
지문:
{}

질문:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<end_of_turn>
<start_of_turn>model
{}<end_of_turn>
"""
BASELINE_RESPONSE_TEMP = "<start_of_turn>model"
BASELINE_END_TURN = "<end_of_turn>"
BASELINE_CHAT_TEMPLETE_PLUS = """지문을 읽고 질문에 대해 단계별로 추론하여 답을 구하세요.<start_of_turn>user
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
{}<end_of_turn>
"""


QWEN_CHAT_TEMPLETE = """<|im_start|>system 
지문을 읽고 질문에 대해 단계별로 추론하여 답을 구하세요.<|im_end|>
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
QWEN_CHAT_TEMPLETE_R = """<|im_start|>system 
지문을 읽고 질문에 대해 단계별로 추론하여 답을 구하세요.<|im_end|>
<|im_start|>user
지문: 
{}

힌트: 
{}

질문: 
{}

선택지: 
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""
QWEN_RESPONSE_TEMP = "<|im_start|>assistant"
QWEN_END_TURN = "<|im_end|>"
QWEN_CHAT_TEMPLETE_PLUS = """<|im_start|>system 
지문을 읽고 질문에 대해 단계별로 추론하여 답을 구하세요.<|im_end|>
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
QWEN_CHAT_TEMPLETE_PLUS_R = """<|im_start|>system 
지문을 읽고 질문에 대해 단계별로 추론하여 답을 구하세요.<|im_end|>
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


LLAMA3_CHAT_TEMPLETE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
지문을 읽고 질문에 대해 단계별로 추론하여 답을 구하세요.<|eot_id|><|start_header_id|>user<|end_header_id|>
지문:
{}

질문:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}<|eot_id|>
"""
LLAMA3_RESPONSE_TEMP = "<|start_header_id|>assistant<|end_header_id|>"
LLAMA3_END_TURN = "<|eot_id|>"
LLAMA3_CHAT_TEMPLETE_PLUS = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
지문을 읽고 질문에 대해 단계별로 추론하여 답을 구하세요.<|eot_id|><|start_header_id|>user<|end_header_id|>
지문:
{}

질문:
{}

<보기>:
{}

선택지:
{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}<|eot_id|>
"""


SOLAR_CHAT_TEMPLETE = """지문을 읽고 질문에 대해 단계별로 추로하여 답을 구하세요.<|im_start|>user
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
SOLAR_CHAT_TEMPLETE_PLUS = """지문을 읽고 질문에 대해 단계별로 추로하여 답을 구하세요.<|im_start|>user
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
