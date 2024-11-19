import csv

def save_evaluation_results_with_id(eval_dataset, model, tokenizer, output_file, max_length):
    """
    평가 데이터의 id와 모델 예측 결과를 CSV로 저장합니다.

    Args:
        eval_dataset (Dataset): 평가 데이터셋 (id 포함).
        model (PreTrainedModel): 평가할 모델.
        tokenizer (PreTrainedTokenizer): 모델에 사용된 토크나이저.
        output_file (str): 결과를 저장할 파일 경로.
        max_length (int): 생성할 출력의 최대 길이.
    """
    logger.info("***** Writing Evaluation Results with ID to CSV *****")
    
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # 헤더 작성
        writer.writerow(["id", "true_output", "model_output"])
        
        for example in eval_dataset:
            # 평가 데이터의 ID 가져오기
            data_id = example["id"]  # 'id' 열에 접근
            
            # 정답 텍스트 디코딩
            true_output = tokenizer.decode(
                example["labels"], skip_special_tokens=True
            )
            
            # 모델 출력 생성
            model_output_ids = model.generate(
                input_ids=torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device),
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            model_output = tokenizer.decode(
                model_output_ids[0], skip_special_tokens=True
            )
            
            # 결과를 CSV 파일에 저장
            writer.writerow([data_id, true_output, model_output])
            logger.info(f"ID: {data_id}, True: {true_output}, Predicted: {model_output}")
