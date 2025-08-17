import pandas as pd
import numpy as np
import os
import json
from matplotlib import pyplot as plt

# Các tham số chính
majors = ["MCS", "DS", "AM"]
kinds_of_qa = ["closed_end", "opened_end", "multihop2"]
tmpl_dirpath = "output_logs/{major}_{kind_of_qa}_{postfix}"
postfix = "hcmus_contriever_standard_rag"

# 1. Đọc dữ liệu Simple RAG từ JSON và mapping với case_std_id
simple_case_results = {f"TH #{i}": {"Simple RAG": []} for i in range(1, 11)}

for major in majors:
    for kind_of_qa in kinds_of_qa:
        # Đọc kết quả Simple RAG
        path = tmpl_dirpath.format(major=major, kind_of_qa=kind_of_qa, postfix=postfix)
        results_fpath = os.path.join(path, "results.json")
        if not os.path.exists(results_fpath):
            print(f"⚠️ Missing: {results_fpath}, skipping...")
            continue
        
        # Đọc file QA data để mapping case_std_id
        qa_data_path = f"../QA for testing/{major}/{major}_{kind_of_qa}.json"
        if not os.path.exists(qa_data_path):
            print(f"⚠️ Missing QA data: {qa_data_path}, skipping...")
            continue
        
        # Đọc dữ liệu Simple RAG
        with open(results_fpath, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Lấy scores từ gpt_judgement
        scores = results.get("gpt_judgement", [])
        
        # Đọc QA data
        with open(qa_data_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        
        # Mapping scores với case_std_id
        for i, (score, qa_item) in enumerate(zip(scores, qa_data)):
            case_std_id = qa_item.get('case_std_id', 1)
            
            # Xử lý score
            if isinstance(score, str):
                if score.startswith("Score: "):
                    score_value = float(score.replace("Score: ", ""))
                elif ". " in score:
                    score_value = float(score.replace(". ", "."))
                else:
                    score_value = float(score)
            else:
                score_value = float(score)
            
            # Thêm vào kết quả
            if 1 <= case_std_id <= 10:
                simple_case_results[f"TH #{case_std_id}"]["Simple RAG"].append(score_value)

# 2. Đọc dữ liệu Graph-based RAG từ CSV và mapping với case_std_id
csv_data = pd.read_csv("gpt_evaluation_results.csv")
graph_case_results = {f"TH #{i}": {"Graph-based RAG": []} for i in range(1, 11)}

for major in majors:
    for kind_of_qa in kinds_of_qa:
        # Lọc dữ liệu CSV theo major và kind_of_qa
        major_data = csv_data[
            (csv_data["dataset"] == major) & 
            (csv_data["test_type"] == kind_of_qa) &
            (csv_data["evaluation_status"] == "success")
        ]
        
        if major_data.empty:
            print(f"⚠️ No data for {major}_{kind_of_qa}")
            continue
        
        # Đọc file QA data để mapping case_std_id
        qa_data_path = f"../QA for testing/{major}/{major}_{kind_of_qa}.json"
        if not os.path.exists(qa_data_path):
            print(f"⚠️ Missing QA data: {qa_data_path}, skipping...")
            continue
        
        with open(qa_data_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        
        # Tạo mapping từ question_id đến case_std_id
        question_to_case = {}
        for qa_item in qa_data:
            question_to_case[qa_item["id"]] = qa_item["case_std_id"]
        
        # Mapping kết quả Graph-based RAG với case_std_id
        for _, row in major_data.iterrows():
            question_id = row.get("question_id", "")
            if question_id in question_to_case:
                case_id = question_to_case[question_id]
                case_name = f"TH #{case_id}"
                if case_name in graph_case_results:
                    graph_case_results[case_name]["Graph-based RAG"].append(row.get("gpt_score", 0))

# 3. Gộp dữ liệu và chuẩn bị cho vẽ biểu đồ
all_case_results = {}
for case_name in simple_case_results.keys():
    all_case_results[case_name] = {
        "Simple RAG": simple_case_results[case_name]["Simple RAG"],
        "Graph-based RAG": graph_case_results[case_name]["Graph-based RAG"]
    }

# 4. Vẽ boxplot với cách vẽ và màu sắc như yêu cầu
fontsize = 26
case_studies = list(all_case_results.keys())
methods = ['Simple RAG', 'Graph-based RAG']
n_case_studies = len(case_studies)
n_methods = len(methods)
width = 0.25

fig, ax = plt.subplots(figsize=(2.5*n_case_studies, 6))

positions = []
data = []
for i, cs in enumerate(case_studies):
    for j, method in enumerate(methods):
        positions.append(i + j*width)
        data.append(all_case_results[cs][method])

# Plot each box
for i in range(n_case_studies):
    for j, method in enumerate(methods):
        pos = i + j*width
        ax.boxplot(
            all_case_results[case_studies[i]][method],
            positions=[pos],
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=['#1f77b4', '#ff7f0e'][j]),
            medianprops=dict(color='black')
        )

# Set x-ticks in the center of each group
group_centers = [i + width/2 for i in range(n_case_studies)]
ax.set_xticks(group_centers)
ax.set_xticklabels(case_studies, rotation=0, fontsize=fontsize)
ax.set_ylabel('Score', fontsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Legend
handles = [plt.Rectangle((0,0),1,1,facecolor=c) for c in ['#1f77b4', '#ff7f0e']]
ax.legend(handles, methods, fontsize=fontsize, loc='lower center', ncol=2)

plt.tight_layout()
plt.savefig("../results/case_study_grouped_boxplot.pdf", dpi=300, bbox_inches='tight')
plt.show()

# 5. In thống kê chi tiết
print("\n" + "="*80)
print("THỐNG KÊ CHI TIẾT THEO CASE STUDIES")
print("="*80)

for case_name in case_studies:
    print(f"\n{case_name}:")
    for method in methods:
        data = all_case_results[case_name][method]
        if data:
            print(f"  {method}:")
            print(f"    Count: {len(data)}")
            print(f"    Mean: {np.mean(data):.3f}")
            print(f"    Std: {np.std(data):.3f}")
            print(f"    Min: {np.min(data):.3f}")
            print(f"    Max: {np.max(data):.3f}")
        else:
            print(f"  {method}: No data")

# 6. Lưu kết quả vào CSV để phân tích thêm
results_df = []
for case_name in case_studies:
    for method in methods:
        data = all_case_results[case_name][method]
        if data:
            results_df.append({
                "case_study": case_name,
                "method": method,
                "count": len(data),
                "mean": np.mean(data),
                "std": np.std(data),
                "min": np.min(data),
                "max": np.max(data)
            })

if results_df:
    results_df = pd.DataFrame(results_df)
    results_df.to_csv("../results/case_studies_statistics.csv", index=False)
    print(f"\n📊 Đã lưu thống kê vào: ../results/case_studies_statistics.csv")

print("\n✅ Hoàn thành vẽ boxplot theo case studies!") 