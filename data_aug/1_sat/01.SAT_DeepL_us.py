import pandas as pd
import requests


def translate_dataset(api_key, input_file, output_file, subject_filter):
    """
    주어진 데이터셋의 특정 과목 데이터를 번역하여 저장합니다.

    Parameters:
        api_key (str): DeepL API 키
        input_file (str): 입력 CSV 파일 경로
        output_file (str): 출력 CSV 파일 경로
        subject_filter (str): 번역할 과목 필터
    """

    # DeepL API URL
    url = "https://api-free.deepl.com/v2/translate"

    # 번역 함수
    def translate_text(text, target_lang="KO"):
        if pd.isna(text):  # 텍스트가 없는 경우 처리
            return None

        params = {"auth_key": api_key, "text": text, "target_lang": target_lang}
        response = requests.post(url, data=params)
        result = response.json()

        # 무료 요금제 한도 초과 시 오류 처리
        if "message" in result and "Quota exceeded" in result["message"]:
            print("번역 한도를 초과했습니다. 더 이상 번역할 수 없습니다.")
            return None  # 번역을 멈추고 None 반환

        return result["translations"][0]["text"]

    # 데이터 불러오기
    df = pd.read_csv(input_file)
    if subject_filter:
        df = df[df["subject"] == subject_filter]

    # 번역 수행
    columns_to_translate = ["prompt", "A", "B", "C", "D", "E"]
    for col in columns_to_translate:
        if col in df.columns:
            print(f"Translating column: {col}")
            df[f"{col}_ko"] = df[col].apply(lambda x: translate_text(x, "KO"))

    # 번역된 데이터 저장
    df.to_csv(output_file, index=False)
    print(f"Translated data saved to {output_file}")


# 스크립트 실행
if __name__ == "__main__":
    # 사용자 입력
    api_key = ""
    input_file = "../resources/sat/sat_world_and_us_history.csv"
    output_file = "../resources/sat/sat_us_history_nonparagraph.csv"

    translate_dataset(api_key, input_file, output_file, subject_filter="us_history")
