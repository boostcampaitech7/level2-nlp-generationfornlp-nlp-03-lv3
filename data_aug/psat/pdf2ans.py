import pdfplumber
import csv
import json
import pandas as pd

TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "explicit_vertical_lines": [],
    "explicit_horizontal_lines": [],
    "snap_tolerance": 3,
    "snap_x_tolerance": 2,
    "snap_y_tolerance": 2,
    "join_tolerance": 3,
    "join_x_tolerance": 3,
    "join_y_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
    "intersection_x_tolerance": 3,
    "intersection_y_tolerance": 3,
    "text_tolerance": 3,
    "text_x_tolerance": 3,
    "text_y_tolerance": 3,
}


def save_table_to_csv(table_data, file_name):
    with open(file_name, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerows(table_data)

def extract_data_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages
        index = 1

        for page in pages:
            tables = page.find_tables(table_settings=TABLE_SETTINGS)
            for i, table in enumerate(tables):
                table_data = table.extract()
                if index==2:
                    if i+1==3 or i+1==4:
                        file_name = f"./aug_src/2022/page_{index}_table_{i + 1}.csv" # # TO_DO 년도만
                        save_table_to_csv(table_data, file_name)
                        print(f"Saved table to {file_name}")
            index += 1

extract_data_from_pdf('./aug_src/2022/최종정답(PDF파일).pdf') # TO_DO 읽어올 정답 pdf 파일명


data_1=pd.read_csv("./aug_src/2022/page_2_table_3.csv") # TO_DO 년도만
data_2=pd.read_csv("./aug_src/2022/page_2_table_4.csv") # TO_DO 년도만

total_ans_data=pd.concat((data_1,data_2),axis=0)
total_ans_data=total_ans_data.reset_index(drop=True)
print(len(total_ans_data))
total_ans_data=total_ans_data.rename(columns={'문번':'id','정답':'answer'})
total_ans_data.to_csv("./aug_src/2022/answer.csv",index=False) # TO_DO 저장할 파일명,경로명