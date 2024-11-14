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
{}<end_of_turn>
"""
BASELINE_RESPONSE_TEMP = "<start_of_turn>model"
BASELINE_END_TURN = "<end_of_turn>"

EXAONE_CHAT_TEMPLETE = """[|system|] 당신은 대한민국 수능 전문가입니다. 수능 국어와 사회 지문, 지문과 관련된 질문, 질문에 대한 선택지가 주어졌을 때 정답을 맞춰주세요.
### 다음 지침을 따라주세요.
1. 지문과 질문을 보고 당신이 알고 있는 사전지식을 떠올려주세요.
2. 사전지식과 선택지를 보고 정답 선택지를 골라주세요.[|endofturn|]
[|user|] 
지문: {}

질문 : {}

선택지:{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:[|endofturn|]
[|assistant|]
{}[|endofturn|]"""
EXAONE_RESPONSE_TEMP = "[|assistant|]"
EXAONE_END_TURN = "[|endofturn|]"

QWEN_CHAT_TEMPLETE = """<|im_start|>system
당신은 대한민국 수능 전문가입니다. 수능 국어와 사회 지문, 지문과 관련된 질문, 질문에 대한 선택지가 주어졌을 때 정답을 맞춰주세요.
### 다음 지침을 따라주세요.
1. 지문과 질문을 보고 당신이 알고 있는 사전지식을 떠올려주세요.
2. 사전지식과 선택지를 보고 정답 선택지를 골라주세요.<|im_end|>
<|im_start|>user
지문: {}

질문 : {}

선택지:{}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""
QWEN_RESPONSE_TEMP = "<|im_start|>assistant"
QWEN_END_TURN = "<|im_end|>"
