# projectmission2/src/data_processing/pdf/pdf_analyzer.py

from typing import List, Dict

def analyze_chunks(chunks: List[Dict]) -> None:
    type_counts, subtype_counts, priority_counts = {}, {}, {}
    for c in chunks:
        type_counts[c['type']]=type_counts.get(c['type'],0)+1
        subtype_counts[c['subtype']]=subtype_counts.get(c['subtype'],0)+1
        p = c['metadata']['priority']
        priority_counts[p]=priority_counts.get(p,0)+1

    print("\n=== 청크 분석 ===")
    print("타입별:", type_counts)
    print("세부타입별:", subtype_counts)
    print("우선순위별:", priority_counts)