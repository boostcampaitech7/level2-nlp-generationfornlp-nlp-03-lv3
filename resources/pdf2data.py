import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import pandas as pd
# from tqdm import tqdm


def extract_text_by_side(page_layout, side="left"):
    """
    페이지 레이아웃에서 지정된 면(왼쪽 또는 오른쪽)의 텍스트를 추출합니다.
    """
    elements = []
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            text = element.get_text()
            if side == "left" and element.x0 < page_layout.width / 2:
                # 왼쪽 페이지 상단 불필요 텍스트 제거
                text = re.sub(r'2024년도\s+국가공무원\s+5급\s+공채\s+등\s+필기시험', '', text)
                elements.append(re.sub(r"\n", "", text))
            elif side == "right" and element.x0 >= page_layout.width / 2:
                # 오른쪽 페이지 상단 불필요 텍스트 제거
                text = re.sub(r'언어논리영역|책형|[0-9]+쪽', '', text)
                elements.append(re.sub(r"\n", "", text))
    return ''.join(elements)


def detect_problem_type(page_text):
    """
    텍스트에서 문제의 유형을 탐지합니다 (단일 문제 또는 묶음 문제).
    """
    try:
        # 묶음 문제 감지
        match = re.search(r"\[(\d+)\s*～\s*(\d+)\]", page_text)
        if match:
            return "grouped"  # 묶음 문제로 간주하여 스킵
        else:
            # 단일 문제
            return "single"
    except Exception as e:
        print(f"[Error] 문제 유형을 탐지하는 중 오류 발생: {e}")
        return "unknown"


def process_single_problem(problem_text):
    """
    단일 문제에서 데이터를 추출합니다.
    """
    try:
        # 질문 추출
        question_match = re.search(r'.*?\?', problem_text)
        # print(question_match)
        if question_match:
            index,question=question_match.group().split(".")

            index=index.strip()
            question=question.strip()
            context_start=question_match.end()
            context_choices_split=re.split(r"(?=\n?[①-⑤]\s)",problem_text[context_start:],maxsplit=1)
            context = context_choices_split[0].strip()
            choices_text=context_choices_split[1]
            choices = re.split(r"(?=[①-⑤])", choices_text)
            choices = [re.sub(r"^[①-⑤]\s?", "", choice.strip()) for choice in choices if choice.strip()]
            return{
                "Index": int(index),
                "Question": question,
                "Context": context,
                "Choices": choices
            }
    except Exception as e:
        print(f"[Error] 단일 문제를 처리하는 중 오류 발생: {e}")
        return None


def extract_page_data(file_path, side="left"):
    """
    지정된 PDF의 특정 면(왼쪽 또는 오른쪽)에서 데이터를 추출하여 DataFrame으로 변환합니다.
    """
    data = []

    for page_layout in extract_pages(file_path):
        try:
            # 페이지 텍스트 추출
            page_text = extract_text_by_side(page_layout, side)
            # 빈 텍스트 처리
            if not page_text.strip():
                print("[Warning] 페이지 텍스트가 비어 있습니다. 건너뜁니다.")
                continue

            # 문제 유형 탐지
            problem_type = detect_problem_type(page_text)

            if problem_type == "grouped":
                print("[Skip] 묶음 문제 페이지를 건너뜁니다.")
                continue  # 묶음 문제는 건너뜁니다.
            elif problem_type == "single":
                single_problem_data = process_single_problem(page_text)
                if single_problem_data:
                    # print(f"단일 문제 처리 완료: {single_problem_data}")
                    data.append(single_problem_data)

        except Exception as e:
            print(f"[Error] 페이지 처리 중 오류 발생: {e}")
    print(f"전체 데이터 개수: {len(data)}")
    return pd.DataFrame(data)



# 사용 예시
file_path = "./aug_data/psat_언어논리_2024.pdf"
left_data = extract_page_data(file_path, side="left")
right_data = extract_page_data(file_path, side="right")


# 출력 예시
total_data = pd.concat((left_data, right_data), axis=0)
total_data = total_data.sort_values(by=['Index'])
total_data = total_data.dropna(subset=['Context']).reset_index(drop=True)
total_data.to_csv("extract_from_pdf.csv", encoding='utf-8-sig')
