import argparse
import json
import os
import sys

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

sys.setrecursionlimit(2000)

LOG_TOP_BOTTOM = False

def calculate_metrics(data, args):
    for i in data:
        if ' ' not in i['original_prediction']:
            i['original_prediction'] += ' '
    bleu_scores = []
    rouge1s = []
    rouge2s = []
    rougels = []
    rouge_l_with_index = []  # Store rouge-l scores with their indices
    rouge = Rouge()
    for index in tqdm(range(len(data)), desc='Calculate BLEU&ROUGE Score'):
        score = sentence_bleu([data[index]['answer']], data[index]['original_prediction'])
        bleu_scores.append(score)
        scores = rouge.get_scores(data[index]['original_prediction'], data[index]['answer'])
        rouge1s.append(scores[0]['rouge-1']['r'])
        rouge2s.append(scores[0]['rouge-2']['r'])
        rouge_l_score = scores[0]['rouge-l']['r']
        rougels.append(rouge_l_score)
        rouge_l_with_index.append((index, rouge_l_score))
    
    # Get top and bottom 5 ROUGE-L scores
    sorted_rouge_l = sorted(rouge_l_with_index, key=lambda x: x[1], reverse=True)
    top5_rouge_l = sorted_rouge_l[:5]
    bottom5_rouge_l = sorted_rouge_l[-5:]
    
    temp_original = {}
    temp_original['BLEU SCORE'] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    temp_original['ROUGE-1'] = sum(rouge1s) / len(rouge1s) if rouge1s else 0
    temp_original['ROUGE-2'] = sum(rouge2s) / len(rouge2s) if rouge2s else 0
    temp_original['ROUGE-L'] = sum(rougels) / len(rougels) if rougels else 0
    
    # Add top and bottom ROUGE-L samples
    if LOG_TOP_BOTTOM:
        temp_original['top5_rouge_l_samples'] = [
            {"index": idx, "score": score, "answer": data[idx]['answer'], "prediction": data[idx]['original_prediction']}
            for idx, score in top5_rouge_l
        ]
        temp_original['bottom5_rouge_l_samples'] = [
            {"index": idx, "score": score, "answer": data[idx]['answer'], "prediction": data[idx]['original_prediction']}
            for idx, score in bottom5_rouge_l
        ]

    # cal bert score
    print("***********Calculate BERT Similarity Score**************")
    sentences1 = [i['answer'] for i in data]
    sentences2 = [i['original_prediction'] for i in data]
    model = SentenceTransformer(args.model_path, device=f"cuda:{args.device}")

    embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
        # print(cosine_scores.shape)
    temp_original['Bert Score'] = cosine_scores.diagonal().mean().item()
    return temp_original

def run_eval(args):
    args = parser.parse_args()

    base_file_path = os.path.basename(args.file_path).lower()
    if 'unke' in base_file_path:
        ds_name = 'unke'
    elif 'cf' in base_file_path:
        ds_name = 'cf'
    elif 'editevery' in base_file_path:
        ds_name = 'editevery'
    elif 'mquake' in base_file_path:
        ds_name = 'mquake'
    elif 'hallucination' in base_file_path:
        ds_name = 'hallucination'
    else:
        raise ValueError(f"Invalid dataset name: {args.file_path}")
    
    with open(args.file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    # data = [i for i in data if 'sub_pred' in i.keys()]
    

    if ds_name == "editevery":
        category_data = {}
        for item in data:
            category = item['category']
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(item)

        matrics = {}
        for category, cat_data in category_data.items():
            print(f"Calculating metrics for category: {category}")
            metrics = calculate_metrics(cat_data, args)
            matrics[category] = metrics

        # print("***********Result**************")
        # print(json.dumps(matrics, indent=4))
        # # save to json
        # import pdb; pdb.set_trace()
        
    elif ds_name == "hallucination":
        category_data = {}
        for item in data:
            category = 'hallucination'
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(item)

        matrics = {}
        for category, cat_data in category_data.items():
            print(f"Calculating metrics for category: {category}")
            metrics = calculate_metrics(cat_data, args)
            matrics[category] = metrics
        
    else:
        for i in data:
            if ' ' not in i['original_prediction']:
                i['original_prediction'] += ' '
            if ds_name in ['unke','cf'] and ' ' not in i['para_prediction']:
                i['para_prediction'] += ' '
            for j in range(len(i['sub_pred'])):
                if ' ' not in i['sub_pred'][j]:
                    i['sub_pred'][j] += ' '

            # if i['original_prediction'] == '':
            #     i['original_prediction'] = ' '
            # if i['para_prediction'] == '':
            #     i['para_prediction'] = ' '
        matrics = {}

        # cal bleu
        bleu_scores = []
        rouge1s=[]
        rouge2s=[]
        rougels=[]
        rouge_l_with_index = []  # Store rouge-l scores with their indices
        
        bleu_scores_para = []
        rouge1s_para=[]
        rouge2s_para=[]
        rougels_para=[]
        rouge_l_para_with_index = []  # Store para rouge-l scores with their indices
        
        rougels_sub=[]
        rouge1s_sub=[]
        rouge2s_sub=[]
        rouge_l_sub_with_index = []  # Store sub rouge-l scores with their indices
        
        rouge = Rouge()

        for index in tqdm(range(len(data)),desc='Calculate BLEU&ROUGE Score'):
            score = sentence_bleu([data[index]['answer']], data[index]['original_prediction'])
            bleu_scores.append(score)
            if ds_name in ['unke','cf']:
                score = sentence_bleu([data[index]['answer']], data[index]['para_prediction'])
                bleu_scores_para.append(score)
            scores = rouge.get_scores(data[index]['original_prediction'],data[index]['answer'])
            rouge1s.append(scores[0]['rouge-1']['r'])
            rouge2s.append(scores[0]['rouge-2']['r'])
            rouge_l_score = scores[0]['rouge-l']['r']
            rougels.append(rouge_l_score)
            rouge_l_with_index.append((index, rouge_l_score))
            if ds_name in ['unke','cf']:
                scores = rouge.get_scores(data[index]['para_prediction'],data[index]['answer'])
                rouge1s_para.append(scores[0]['rouge-1']['r'])
                rouge2s_para.append(scores[0]['rouge-2']['r'])
                rouge_l_para_score = scores[0]['rouge-l']['r']
                rougels_para.append(rouge_l_para_score)
                rouge_l_para_with_index.append((index, rouge_l_para_score))
            sub_ls = 0
            sub_1s = 0
            sub_2s = 0
            for i in range(len(data[index]['sub_pred'])):
                scores = rouge.get_scores(data[index]['sub_pred'][i],data[index]['sub_answer'][i])
                sub_1s += scores[0]['rouge-1']['r']
                sub_2s += scores[0]['rouge-2']['r']
                sub_ls += scores[0]['rouge-l']['r']
            avg_sub_ls = sub_ls/len(data[index]['sub_pred'])
            rouge1s_sub.append(sub_1s/len(data[index]['sub_pred']))
            rouge2s_sub.append(sub_2s/len(data[index]['sub_pred']))
            rougels_sub.append(avg_sub_ls)
            rouge_l_sub_with_index.append((index, avg_sub_ls))

        
        # Get top and bottom 5 ROUGE-L scores for Original
        sorted_rouge_l = sorted(rouge_l_with_index, key=lambda x: x[1], reverse=True)
        top5_rouge_l = sorted_rouge_l[:5]
        bottom5_rouge_l = sorted_rouge_l[-5:]
        
        # Get top and bottom 5 ROUGE-L scores for Para (if applicable)
        if ds_name in ['unke', 'cf']:
            sorted_rouge_l_para = sorted(rouge_l_para_with_index, key=lambda x: x[1], reverse=True)
            top5_rouge_l_para = sorted_rouge_l_para[:5]
            bottom5_rouge_l_para = sorted_rouge_l_para[-5:]
        
        # Get top and bottom 5 ROUGE-L scores for Sub
        sorted_rouge_l_sub = sorted(rouge_l_sub_with_index, key=lambda x: x[1], reverse=True)
        top5_rouge_l_sub = sorted_rouge_l_sub[:5]
        bottom5_rouge_l_sub = sorted_rouge_l_sub[-5:]
        
        temp_original = {}
        temp_para = {}
        temp_sub={}
        temp_original['BLEU SCORE'] = sum(bleu_scores) / len(bleu_scores)
        temp_original['ROUGE-1'] = sum(rouge1s) / len(rouge1s)
        temp_original['ROUGE-2'] = sum(rouge2s) / len(rouge2s)
        temp_original['ROUGE-L'] = sum(rougels) / len(rougels)
        
        # Add top and bottom ROUGE-L samples for Original
        if LOG_TOP_BOTTOM:
            temp_original['top5_rouge_l_samples'] = [
                {"index": idx, "score": score, "answer": data[idx]['answer'], "prediction": data[idx]['original_prediction']}
                for idx, score in top5_rouge_l
            ]
            temp_original['bottom5_rouge_l_samples'] = [
                {"index": idx, "score": score, "answer": data[idx]['answer'], "prediction": data[idx]['original_prediction']}
                for idx, score in bottom5_rouge_l
            ]
        
        if ds_name in ['unke','cf']:
            temp_para['BLEU SCORE'] = sum(bleu_scores_para) / len(bleu_scores_para)
            temp_para['ROUGE-1'] = sum(rouge1s_para) / len(rouge1s_para)
            temp_para['ROUGE-2'] = sum(rouge2s_para) / len(rouge2s_para)
            temp_para['ROUGE-L'] = sum(rougels_para) / len(rougels_para)
            
            # Add top and bottom ROUGE-L samples for Para
            if LOG_TOP_BOTTOM:
                temp_para['top5_rouge_l_samples'] = [
                    {"index": idx, "score": score, "answer": data[idx]['answer'], "prediction": data[idx]['para_prediction']}
                    for idx, score in top5_rouge_l_para
                ]
                temp_para['bottom5_rouge_l_samples'] = [
                    {"index": idx, "score": score, "answer": data[idx]['answer'], "prediction": data[idx]['para_prediction']}
                    for idx, score in bottom5_rouge_l_para
                ]

        temp_sub['ROUGE-1'] = sum(rouge1s_sub) / len(rouge1s_sub)
        temp_sub['ROUGE-2'] = sum(rouge2s_sub) / len(rouge2s_sub)
        temp_sub['ROUGE-L'] = sum(rougels_sub) / len(rougels_sub)
        
        # Add top and bottom ROUGE-L samples for Sub
        if LOG_TOP_BOTTOM:
            temp_sub['top5_rouge_l_samples'] = [
                {"index": idx, "score": score, 
                "sub_answers": data[idx]['sub_answer'], 
                "sub_predictions": data[idx]['sub_pred']}
                for idx, score in top5_rouge_l_sub
            ]
            temp_sub['bottom5_rouge_l_samples'] = [
                {"index": idx, "score": score, 
                "sub_answers": data[idx]['sub_answer'], 
                "sub_predictions": data[idx]['sub_pred']}
                for idx, score in bottom5_rouge_l_sub
            ]
        # cal bert score
        print("***********Calculate BERT Similarity Score**************")
        sentences1 = [i['answer'] for i in data]
        sentences2 = [i['original_prediction'] for i in data]
        if ds_name in ['unke','cf']:
            sentences3 = [i['para_prediction'] for i in data]
        model = SentenceTransformer(args.model_path, device=f"cuda:{args.device}")

        embeddings1 = model.encode(sentences1, convert_to_tensor=True,show_progress_bar=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True,show_progress_bar=True)
        if ds_name in ['unke','cf']:
            embeddings3 = model.encode(sentences3, convert_to_tensor=True,show_progress_bar=True)
        # Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        # print(cosine_scores.shape)
        temp_original['Bert Score'] = cosine_scores.diagonal().mean().item()
        temp_bert_score = cosine_scores.diagonal().cpu().numpy().tolist()

        if ds_name in ['unke','cf']:
            cosine_scores = util.cos_sim(embeddings1, embeddings3)
            # print(cosine_scores.shape)
            temp_para['Bert Score'] = cosine_scores.diagonal().mean().item()
            temp_bert_score_para = cosine_scores.diagonal().cpu().numpy().tolist()
        matrics['Original']=temp_original
        if ds_name in ['unke','cf']:
            matrics['Para']=temp_para
        matrics['Sub']=temp_sub
        # temp_result = [bleu_scores,bleu_scores_para,rouge1s,rouge1s_para,rouge2s,rouge2s_para,rougels,rougels_para,temp_bert_score,temp_bert_score_para]
        # with open('data_memit.json', 'w') as file:
        #     json.dump(temp_result, file)
    print("***********Result**************")
        # print(args.file_path)
        # print(f'{".".join(args.file_path.split("/")[-1].split(".")[:2])}.json')
        # save to json
        # args.file_path.split('/')
        # ['output', 'unke_ARE-Qwen', 'Qwen2.5-7B-Instruct-unke-100-step=25-NE=4_result.json']
        
    # Extract the directory and filename parts
    path_parts = args.file_path.split('/')
    if len(path_parts) >= 2:
        model_dir = path_parts[-2]  # e.g., 'unke_ARE-Qwen'
        output_dir = f'output/summarize_uns/{model_dir}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{".".join(path_parts[-1].split(".")[:2])}.json'
    else:
        # Fallback if path structure is unexpected
        output_dir = 'output/summarize_uns'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{".".join(path_parts[-1].split(".")[:2])}.json'
        
    with open(output_path, 'w') as file:
        json.dump(matrics, file, indent=4)

    # *100 for each metric
    for key in matrics:
        for metric in matrics[key]:
            matrics[key][metric] *= 100
    print(json.dumps(matrics, indent=4))
    # Print flattened CSV format
    csv_line = []
    if ds_name == 'editevery':
        for key in matrics:
            if key in matrics:
                metrics_order = ['BLEU SCORE', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Bert Score'] 
                for metric in metrics_order:
                    if metric in matrics[key]:
                        csv_line.append(f"{matrics[key][metric]}")
    else:
        for key in ['Original', 'Para', 'Sub']:
            if key in matrics:
                metrics_order = ['BLEU SCORE', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Bert Score'] 
                for metric in metrics_order:
                    if metric in matrics[key]:
                        csv_line.append(f"{matrics[key][metric]}")
    print(','.join(csv_line))
    return matrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='output/unke_Llama3-8B-Instruct_cf_result.json', type=str)
    parser.add_argument('--model_path', default='sentence-transformers/all-MiniLM-L6-v2', type=str)
    parser.add_argument('--device', default=0, type=int)

    run_eval(parser.parse_args())