# Text Summarizer Project

An end-to-end **NLP Text Summarization** project using **Google PEGASUS** (`google/pegasus-cnn_dailymail`), fine-tuned on the **SAMSum** dataset. Built with a modular, production-ready architecture and served via a **FastAPI** web application.

## ✨ Features

- **5-Stage ML Pipeline**: Data Ingestion → Validation → Transformation → Training → Evaluation
- **Google PEGASUS Model**: State-of-the-art abstractive summarization
- **FastAPI Web Interface**: Clean, modern UI to summarize text in real-time
- **Modular Architecture**: Each component is independently configurable via YAML
- **Docker Support**: Ready for containerized deployment
- **ROUGE Evaluation**: Automated model evaluation with ROUGE-1, ROUGE-2, ROUGE-L metrics

## 📁 Project Structure

```
Text-Summarizer-Project/
├── config/
│   └── config.yaml                  # Pipeline configuration (paths, URLs, model names)
├── src/textSummarizer/
│   ├── components/                  # ML components (ingestion, validation, transformation, training, evaluation)
│   ├── config/configuration.py      # Configuration manager
│   ├── constants/                   # File path constants
│   ├── entity/                      # Typed config dataclasses
│   ├── logging/                     # Custom logger
│   ├── pipeline/                    # Pipeline stages + prediction
│   └── utils/common.py             # Utility functions
├── templates/
│   └── index.html                   # Web UI template
├── research/
│   └── trials.ipynb                 # Experimentation notebook
├── app.py                           # FastAPI web application
├── main.py                          # Training pipeline entry point
├── params.yaml                      # Training hyperparameters
├── Dockerfile                       # Container configuration
├── setup.py                         # Package setup
└── requirements.txt                 # Python dependencies
```

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Training Pipeline
```bash
python main.py
```
This runs all 5 stages: data download → validation → tokenization → training → evaluation.

### 3. Start the Web Application
```bash
python app.py
```
Open your browser to `http://localhost:8080`.

### 4. Run on Google Colab
Upload the project to Google Drive, or clone from GitHub, then run `main.py` in a Colab notebook with GPU runtime enabled.

## 🐳 Docker

```bash
docker build -t text-summarizer .
docker run -p 8080:8080 text-summarizer
```

## 📊 Model & Dataset

| Item | Details |
|------|---------|
| **Base Model** | [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail) |
| **Dataset** | [SAMSum](https://huggingface.co/datasets/samsum) (16k dialogue-summary pairs) |
| **Metrics** | ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum |

## 📥 Download & Setup Pre-trained Weights

Since the fine-tuned model weights are large (2.2GB), they are not included in this repository. Follow these steps to set them up correctly:

### 1. Download
Download all files from this folder:
- **[Download Fine-tuned Weights (Google Drive)](https://drive.google.com/drive/folders/1JqetHgteEGAB-D1fzPob1yuUrRl06ZXJ?usp=drive_link)**

### 2. Organize & "Merge" Files
You need to create a specific folder structure inside the project. Follow these exact steps:

1.  Create the following folders in your project root:
    `artifacts/model_trainer/pegasus-samsum-model`
    `artifacts/model_trainer/tokenizer`

2.  **Move the files** as follows:
    *   Place `config.json` and `generation_config.json` into:
        `artifacts/model_trainer/pegasus-samsum-model/`
    *   Place the large `model-001.safetensors` into:
        `artifacts/model_trainer/pegasus-samsum-model/` and **rename it** to `model.safetensors`.
    *   Place `tokenizer.json` and `tokenizer_config.json` into:
        `artifacts/model_trainer/tokenizer/`

Your structure should look like this:
```text
artifacts/
└── model_trainer/
    ├── pegasus-samsum-model/
    │   ├── config.json
    │   ├── generation_config.json
    │   └── model.safetensors       <-- (Renamed from model-001.safetensors)
    └── tokenizer/
        ├── tokenizer.json
        └── tokenizer_config.json
```

Once this is done, `python app.py` will automatically detect and use your fine-tuned model!

## 📈 Comparison of Summarization Models

| Model | Developed By | Best Use Case | Approx. Size |
|-------|-------------|---------------|--------------|
| **PEGASUS** | Google | Specifically designed for abstractive summarization. Specialized in news and dialogue. | **2.2 GB** |
| **BART** | Facebook | Excellent for conversational data and informal text. Very strong on SAMSum. | **1.6 GB** |
| **T5** | Google | Versatile text-to-text model. Balanced efficiency and performance. | **240MB - 900MB** |
| **GPT/Llama** | OpenAI / Meta | State-of-the-art general purpose LLMs. Highest quality but very large. | **15GB+** |

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
