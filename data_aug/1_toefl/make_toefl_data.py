import os
import pandas as pd


def load_data(data_path):
    train_path = os.path.join(data_path, "train")
    dev_path = os.path.join(data_path, "dev")
    test_path = os.path.join(data_path, "test")

    train_data = load_data_workhorse(train_path)
    dev_data = load_data_workhorse(dev_path)
    test_data = load_data_workhorse(test_path)

    return [train_data, dev_data, test_data]


def load_data_workhorse(type_path):
    data = {}
    for each_file in os.listdir(type_path):
        data[each_file] = {}
        data[each_file]["sentences"] = []
        data[each_file]["question"] = []
        data[each_file]["options"] = []
        data[each_file]["answer"] = []
        with open(os.path.join(type_path, each_file), "r") as f:
            for each_line in f:
                parsed = each_line.strip().split()
                if parsed[0] == "SENTENCE":
                    data[each_file]["sentences"].append(parsed[1:])
                elif parsed[0] == "QUESTION":
                    data[each_file]["question"] = parsed[1:]
                else:
                    data[each_file]["options"].append(parsed[1:-1])
                    if parsed[-1] == "1":
                        data[each_file]["answer"] = parsed[1:-1]
    return data


def make_structured(loaded_data):
    sample_list = []
    for keys in loaded_data.keys():
        tmp_dict = {}
        tmp_choices = []
        tmp_answer = []
        tmp_question = []
        tmp_sentences = []
        if "conversation" in keys:
            continue

        tmp_dict["id"] = f"{keys}"

        for sentences_idx in range(len(loaded_data[keys]["sentences"])):
            sentences_txt_part = " ".join(loaded_data[keys]["sentences"][sentences_idx])
            tmp_sentences.append(sentences_txt_part)
        tmp_dict["paragraph"] = " ".join(tmp_sentences)

        for question_idx in range(len(loaded_data[keys]["question"])):
            question_txt_part = "".join(loaded_data[keys]["question"][question_idx])
            tmp_question.append(question_txt_part)
        tmp_dict["question"] = " ".join(tmp_question)

        for options_idx in range(len(loaded_data[keys]["options"])):  # choices
            options_txt_part = " ".join(loaded_data[keys]["options"][options_idx])
            tmp_choices.append(options_txt_part)
        tmp_dict["choices"] = tmp_choices

        for answer_idx in range(len(loaded_data[keys]["answer"])):
            answer_txt_part = "".join(loaded_data[keys]["answer"][answer_idx])
            tmp_answer.append(answer_txt_part)
        tmp_dict["answer"] = " ".join(tmp_answer)

        for choice_idx in range(len(tmp_dict["choices"])):
            if tmp_dict["choices"][choice_idx] == tmp_dict["answer"]:
                tmp_dict["answer_number"] = choice_idx + 1
                break
            else:
                tmp_dict["answer_number"] = None

        sample_list.append(tmp_dict)
    return sample_list


if __name__ == "__main__":
    data_path = "../resources/aug/toefl"
    train_data, dev_data, test_data = load_data(data_path)

    structured_train = make_structured(train_data)
    structured_dev = make_structured(dev_data)
    structured_test = make_structured(test_data)

    train_df = pd.DataFrame(structured_train)
    dev_df = pd.DataFrame(structured_dev)
    test_df = pd.DataFrame(structured_test)

    final_df = pd.concat((train_df, dev_df, test_df), axis=0)
    final_df = final_df.drop("answer", axis=1)
    final_df = final_df.rename(columns={"answer_number": "answer"})
    final_df.to_csv("../resources/aug/toefl/toefl_aug.csv", encoding="utf-8-sig", index=False)
