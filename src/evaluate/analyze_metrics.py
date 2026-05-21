import json
import os
from statistics import mean
from pathlib import Path

# Đường dẫn đến thư mục
eval_dir = r'd:\Python\medbotvn-nckh\outputs\evaluation'

# Danh sách metrics cần tính
metrics = ['answer_correctness', 'answer_relevancy', 'answer_completeness', 'medical_safety']

# Dictionary lưu kết quả
results = {}

# Danh sách các file JSONL
files = {
    'B1_Llama_no_rag': 'B1_Llama_no_rag_target400_scores.jsonl',
    'B2_Llama_rag_mmr_k5': 'B2_Llama_rag_mmr_k5_target400_scores.jsonl',
    'B1_Qwen_no_rag': 'B1_Qwen_no_rag_target400_scores.jsonl',
    'B2_Qwen_rag_mmr_k5': 'B2_Qwen_rag_mmr_k5_target400_scores.jsonl'
}

# Đọc từng file
for config_name, filename in files.items():
    filepath = os.path.join(eval_dir, filename)
    if not os.path.exists(filepath):
        print(f'File không tồn tại: {filepath}')
        continue
    
    # Lưu các giá trị metrics
    metrics_data = {metric: [] for metric in metrics}
    
    # Đọc file JSONL
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    for metric in metrics:
                        if metric in record and record[metric] is not None:
                            try:
                                metrics_data[metric].append(float(record[metric]))
                            except (ValueError, TypeError):
                                pass
    except Exception as e:
        print(f'Lỗi khi đọc {filename}: {e}')
        continue
    
    # Tính mean cho từng metric
    results[config_name] = {}
    for metric in metrics:
        if metrics_data[metric]:
            results[config_name][metric] = mean(metrics_data[metric])
        else:
            results[config_name][metric] = 'N/A'

# In kết quả dưới dạng bảng
print('='*80)
print('PHÂN TÍCH METRICS TRUNG BÌNH')
print('='*80)
print()

# In header
header = 'Config Name' + ' '*19 + 'Correctness' + ' '*3 + 'Relevancy' + ' '*5 + 'Completeness' + ' '*2 + 'Medical Safety'
print(header)
print('-'*80)

# In dữ liệu
config_order = ['B1_Llama_no_rag', 'B2_Llama_rag_mmr_k5', 'B1_Qwen_no_rag', 'B2_Qwen_rag_mmr_k5']
for config_name in config_order:
    if config_name in results:
        row_data = results[config_name]
        correctness = f'{row_data["answer_correctness"]:.4f}' if row_data['answer_correctness'] != 'N/A' else 'N/A'
        relevancy = f'{row_data["answer_relevancy"]:.4f}' if row_data['answer_relevancy'] != 'N/A' else 'N/A'
        completeness = f'{row_data["answer_completeness"]:.4f}' if row_data['answer_completeness'] != 'N/A' else 'N/A'
        safety = f'{row_data["medical_safety"]:.4f}' if row_data['medical_safety'] != 'N/A' else 'N/A'
        
        print(f'{config_name:<30} {correctness:<15} {relevancy:<15} {completeness:<15} {safety:<15}')

print()
print('='*80)
print()

# In kết quả chi tiết
print('CHI TIẾT KẾT QUẢ:')
print()
for config_name in config_order:
    if config_name in results:
        print(f'Config: {config_name}')
        print('-' * 40)
        for metric in metrics:
            value = results[config_name][metric]
            if value != 'N/A':
                print(f'  {metric:<30}: {value:.4f}')
            else:
                print(f'  {metric:<30}: {value}')
        print()
