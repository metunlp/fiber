# Do not change anything in this file.

import json
import time
import os

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


@torch.no_grad()
def sequence_logprob(prompt: str, surface: str, model, tokenizer) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    surface_ids = tokenizer.encode(surface[:-1], add_special_tokens=False)
    total = 0.0
    ctx = torch.tensor([prompt_ids], device=model.device)
    T = 0.1
    for j, tok in enumerate(surface_ids):
        out = model(input_ids=ctx)
        last_logits = out.logits[:, -1, :].to(torch.float64)
        logp = F.log_softmax(last_logits / T, dim=-1)[0, tok].item()
        total += logp
        ctx = torch.cat([ctx, torch.tensor([[tok]], device=model.device)], dim=1)
    return total


def precision(gold, all):
    set1 = set(gold)
    set2 = set(all)
    return len(set1.intersection(set2)) / len(set2)


def recall(gold, all):
    set1 = set(gold)
    set2 = set(all)
    return len(set1.intersection(set2)) / len(set1)


def average_precision(ranked_list, relevant_set):
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(ranked_list, start=1):
        if item in relevant_set:
            hits += 1
            precision_at_i = hits / i
            sum_precisions += precision_at_i
    if len(relevant_set) == 0:
        return 0.0
    return sum_precisions / len(relevant_set)


def avg_prec_full(prompt, model, gold, all, tokenizer):
    results = {}
    surface = all
    for i in surface:
        results[i] = sequence_logprob(prompt, i, model, tokenizer)
    sorted_items_desc = sorted(results.items(), key=lambda x: x[1], reverse=True)
    sorted_results = dict(sorted_items_desc)
    prec = precision(gold, all)
    rec = recall(gold, all)
    ap = average_precision(sorted_results.keys(), set(gold))
    return prec, rec, ap, sorted_results


def append_to_json_file(new_data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8-sig") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(new_data)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def test_one_set(data, result_file, folder_name, file_name, main_folder_name, model, tokenizer):
    all = []
    for i in data:
        if main_folder_name == "single_entity":
            all.append(i["answer"])
        if main_folder_name == "multi_entity":
            if file_name.endswith(("en_2.json", "en_3.json", "tr_2.json", "tr_3.json", "it_3.json", "it_2.json")):
                all.append(i["answer"])
            else:
                for x in i["list"]:
                    all.append(x)

    start = time.perf_counter()
    results = {"Subject": [], "Precision": [], "Recall": []}

    for i in data:
        start_1 = time.perf_counter()
        if main_folder_name == "single_entity":
            prec, rec, ap, top_5 = avg_prec_full(i['query'], model, [i['answer']], list(set(all)), tokenizer)
        if main_folder_name == "multi_entity":
            if file_name.endswith(("en_2.json", "en_3.json", "tr_2.json", "tr_3.json", "it_3.json", "it_2.json")):
                prec, rec, ap, top_5 = avg_prec_full(i['query'], model, [i['answer']], list(set(all)), tokenizer)
            else:
                prec, rec, ap, top_5 = avg_prec_full(i['query'], model, i['list'], list(set(all)), tokenizer)
        end_1 = time.perf_counter()
        results["Subject"] = i['sub']
        results["Precision"] = prec
        results["Recall"] = rec
        results["Average precision"] = ap
        results["Top_5"] = top_5
        append_to_json_file(results, f"{result_file}/{folder_name}/{file_name}_results.json")
        print(f"Subject: {i['sub']}, Precision: {prec}, Recall: {rec}, Average precision: {ap}")
        print(f"{i['sub']} {end_1 - start_1}")
    end = time.perf_counter()
    print(f"{file_name}: {end - start}")


def test_one_folder(folder_path, result_path, main_folder_name, model, tokenizer):
    start = time.perf_counter()
    file_paths = [
        os.path.abspath(os.path.join(folder_path, f))
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and f != ".DS_Store"
        and f != ".translation_cache.json"
    ]
    folder_name = os.path.basename(folder_path)
    print(file_paths)
    for i in file_paths:
        file_path = i
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        print(len(data))
        file_name = os.path.basename(i)
        try:
            test_one_set(data, result_path, folder_name, file_name, main_folder_name, model, tokenizer)
        except:
            continue
    end = time.perf_counter()
    print(f"{folder_name}: {end - start}")
    append_to_json_file({"Subject": folder_name, "Time": end - start},
                        f"{result_path}/{folder_name}/{folder_name}_time_results.json")


def test_all_data(folder_path, result_path, model, tokenizer):
    start = time.perf_counter()
    folders = [
        os.path.abspath(os.path.join(folder_path, f))
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]
    print(folders)
    for i in folders:
        print(i)
        folders_2 = [
            os.path.abspath(os.path.join(i, f))
            for f in os.listdir(i)
            if os.path.isdir(os.path.join(i, f))
        ]
        folder_name = os.path.basename(i)
        for x in folders_2:
            print(x)
            test_one_folder(x, result_path, folder_name, model, tokenizer)
    end = time.perf_counter()
    print(f"All data: {end - start}")
    append_to_json_file({"Subject": "All", "Time": end - start},
                        f"{result_path}/time.json")


def main():
    with open('../config.json', 'r', encoding='utf-8-sig') as f:
        config = json.load(f)
    hugging_face_token = config["hugging_face_token"]
    model_id = config["model_id"]
    data_dir = "../" + config["data_dir"]
    result_dir = "../" + config["result_dir"]
    if not os.path.exists(data_dir):
        print("Enter a valid input directory.")
        return
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    login(hugging_face_token)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to("cuda")
    test_all_data(data_dir, result_dir, model, tokenizer)


if __name__ == "__main__":
    main()
