# Synthetic Identity Fraud Detection

This project is built for the IIEST-UCO Bank Hackathon 2025. It aims to accurately detect synthetic identity fraud using a combination of traditional machine learning, rule-based logic, and **graph-based modeling** to link seemingly unrelated fraudulent accounts.

## Features

- **Ensemble Model** combining:
  - Random Forest
  - Isolation Forest
  - Rule-based heuristics
  - Graph-based fraud propagation detection
- **Graph-Based Link Analysis**
  - Connects users through identifiers like `email`, `phone_number`, `ip_address`, and `device_id`
  - Uncovers hidden fraud rings
- **Explainability**
  - Returns fraud score, fraud label, and clear reasons for flagging

---

## Backend Structure

```bash
backend/
├── Dataset/
│   └── (load large datasets from Google Drive - see below)
├── models/
│   └── (load pretrained models from Google Drive)
├── ensemble.py
├── preprocess.py
├── graph_utils.py
├── graph_features.py
├── rules.py
├── main.py  ← API using FastAPI
├── test.py  ← Test sample input
```

## 💾 Download Large Files

Due to GitHub's file size limits, please manually download the following files from Google Drive:

| Folder Name                      | Description                       | Link                  |
|-------------------------------|-----------------------------------|-----------------------|
| `Dataset`                    | Datasets (Base.csv, Base_with_identifiers.csv)                   | [Download](https://drive.google.com/drive/folders/1Kqrri7XLqt9j6Pf4YYNhKLqPyDrSujuc?usp=sharing) |
| `models`                | Trained Models and Files      | [Download](https://drive.google.com/drive/folders/1sRzhb96Mu_AOw9ENy011Vk26w0tg1gSU?usp=sharing) |

> ⚠️ **Important:** After downloading, place the files in the appropriate directories:

## Tech Stack

- **Language**: Python
- **Backend**: FastAPI
- **ML Models**: scikit-learn (Random Forest, Isolation Forest)
- **Graph Analysis**: NetworkX
- **Data Handling**: Pandas, NumPy
- **Serialization**: joblib, pickle

## License

This project is licensed under the MIT License - see the LICENSE file for details.

