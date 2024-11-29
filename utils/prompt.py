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
#----------------------------------------------------------------
KOWIKI_CHAT_TEMPLETE = """<|im_start|>system
문제를 풀기 위해 단계별 사고 과정을 거쳐 정답을 도출하세요.<|im_end|>
<|im_start|>user
학습 목적:
{}

배경 지식:
{}

문제:
{}

사고 흐름:
{}

사고 흐름 답변:
{}

정답을 도출해 주세요.
<|im_end|>
<|im_start|>assistant
{}
<|im_end|>"""

KOWIKI_RESPONSE_TEMP = "<|im_start|>assistant"
KOWIKI_END_TURN = "<|im_end|>"
