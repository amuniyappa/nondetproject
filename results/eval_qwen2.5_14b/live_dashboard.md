# 📊 Agentless Live Evaluation Dashboard

**Model:** `qwen2.5-coder-14b-instruct` | **Backend:** `qwen` | **Base URL:** `http://127.0.0.1:1234/v1`
**Last Updated:** `2026-10-23T23:57:20.507975` | **Tasks Processed:** `23`

## 🎯 Performance Metrics Across All Phases (Paper Align)

| Metric | Paper Phase | Performance | Count |
| :--- | :--- | :--- | :--- |
| **File Localization (Top-1)** | Phase 1 (File FL) | **65.2%** | 15/23 |
| **File Localization (Top-3)** | Phase 1 (File FL) | **78.3%** | 18/23 |
| **File Localization (Top-5)** | Phase 1 (File FL) | **78.3%** | 18/23 |
| **Edit Line Overlap** | Phase 1 (Line FL) | **30.4%** | 7/23 |
| **Valid Git Diff Rate** | Phase 2 (Repair) | **56.5%** | 13/23 |
| **Touches GT Modified File** | Phase 2 (Repair) | **43.5%** | 10/23 |
| **Average Latency per Task** | Efficiency | **232.3s** | Total: 89.0m |
| **Total Tokens Consumed** | Efficiency | **211,119** | P: 207,913 / C: 3,206 |

## 📋 Task-by-Task Real-Time Results

| # | Task ID | File Top-1 | File Top-3 | Line Match | Diff Valid | Touches GT | Tokens | Latency | Status |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 23 | `django__django-11797` | ✗ | ✗ | ✗ | ✗ | ✗ | 5,001 | 290.6s | **FAILED** |
| 22 | `django__django-11742` | ✓ | ✓ | ✗ | ✗ | ✗ | 5,509 | 239.0s | **PARTIAL** |
| 21 | `django__django-11630` | ✗ | ✗ | ✗ | ✗ | ✗ | 4,974 | 254.96s | **FAILED** |
| 20 | `django__django-11620` | ✗ | ✓ | ✗ | ✓ | ✗ | 9,597 | 252.87s | **SUCCESS** |
| 19 | `django__django-11583` | ✓ | ✓ | ✗ | ✓ | ✓ | 11,940 | 238.06s | **SUCCESS** |
| 18 | `django__django-11564` | ✗ | ✗ | ✗ | ✓ | ✗ | 17,105 | 298.73s | **FAILED** |
| 17 | `django__django-11422` | ✓ | ✓ | ✗ | ✓ | ✓ | 10,711 | 226.92s | **SUCCESS** |
| 16 | `django__django-11283` | ✗ | ✓ | ✓ | ✓ | ✓ | 16,040 | 252.94s | **SUCCESS** |
| 15 | `django__django-11179` | ✓ | ✓ | ✗ | ✓ | ✓ | 5,944 | 294.69s | **SUCCESS** |
| 14 | `django__django-11133` | ✓ | ✓ | ✗ | ✓ | ✗ | 10,942 | 246.28s | **SUCCESS** |
| 13 | `django__django-11099` | ✓ | ✓ | ✓ | ✓ | ✓ | 6,151 | 223.49s | **SUCCESS** |
| 12 | `django__django-11049` | ✓ | ✓ | ✓ | ✓ | ✓ | 9,173 | 244.03s | **SUCCESS** |
| 11 | `django__django-11039` | ✗ | ✗ | ✗ | ✗ | ✗ | 6,665 | 233.33s | **FAILED** |
| 10 | `django__django-11019` | ✓ | ✓ | ✗ | ✗ | ✗ | 5,124 | 248.46s | **PARTIAL** |
| 9 | `django__django-11001` | ✓ | ✓ | ✗ | ✗ | ✗ | 10,792 | 241.86s | **PARTIAL** |
| 8 | `django__django-10924` | ✗ | ✗ | ✗ | ✗ | ✗ | 10,710 | 261.6s | **FAILED** |
| 7 | `astropy__astropy-7746` | ✓ | ✓ | ✗ | ✗ | ✗ | 8,052 | 154.6s | **PARTIAL** |
| 6 | `astropy__astropy-6938` | ✓ | ✓ | ✗ | ✓ | ✓ | 5,850 | 165.17s | **SUCCESS** |
| 5 | `astropy__astropy-14995` | ✗ | ✓ | ✓ | ✓ | ✓ | 7,219 | 194.26s | **SUCCESS** |
| 4 | `astropy__astropy-14365` | ✓ | ✓ | ✓ | ✓ | ✓ | 11,604 | 172.58s | **SUCCESS** |
| 3 | `astropy__astropy-14182` | ✓ | ✓ | ✓ | ✗ | ✗ | 17,620 | 194.28s | **PARTIAL** |
| 2 | `astropy__astropy-12907` | ✓ | ✓ | ✗ | ✗ | ✗ | 3,225 | 173.7s | **PARTIAL** |
| 1 | `django__django-10914` | ✓ | ✓ | ✓ | ✓ | ✓ | 11,171 | 240.14s | **SUCCESS** |
