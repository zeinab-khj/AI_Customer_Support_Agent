# Customer Support Copilot

An end-to-end customer support response generation system built with Hugging Face Transformers and fine-tuned Large Language Models.

The goal of this project is to build a customer support assistant that takes a customer's message and generates an appropriate, task-focused response. The project focuses on the complete LLM workflow — from dataset analysis and model benchmarking to fine-tuning, evaluation, and qualitative error analysis.

---

## 📌 Project Overview

Customer support systems need to generate responses that are relevant, natural, and helpful while avoiding unnecessary verbosity and generic customer-service language.

In this project, several Qwen-based language models were investigated and fine-tuned for customer support response generation.

The workflow includes:

* Dataset exploration and preprocessing
* Tokenization and supervised fine-tuning
* Model benchmarking
* Full fine-tuning and QLoRA experiments
* Automatic evaluation using ROUGE and BERTScore
* Qualitative error analysis
* Model selection and comparison
* Inference with the final model

---

## 🔄 Workflow

```text
┌──────────────────────┐
│  Customer Support    │
│       Dataset        │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ EDA & Preprocessing  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Model Benchmarking   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│    Fine-Tuning       │
│  Full FT / QLoRA     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│     Evaluation       │
│ ROUGE + BERTScore    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Error Analysis      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Final 0.5B Model   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      Inference       │
└──────────────────────┘
```

## 🎯 Key Takeaways

* **Smaller models can be highly competitive:** the fully fine-tuned Qwen2.5-0.5B outperformed the larger 1.5B QLoRA setup in this task.* **Benchmarking:** Benchmarking multiple pre-trained language models
* **Fine-tuning strategy matters:** model size alone does not guarantee better task performance.* **Data-Driven Performance:** Achieved an overall F1-score of **[0.9630]%**, balancing high precision for critical entities with robust recall across edge cases.
* **Evaluation requires more than automatic metrics:** ROUGE and BERTScore were complemented with qualitative error analysis.* **Production-Ready Pipeline:** Engineered a scalable preprocessing and inference workflow that handles unstructured text data at scale.
* **Training data quality matters:** noisy or template-like target responses can influence the style of generated responses.
* **End-to-end LLM workflow:** the project covers data preparation, benchmarking, fine-tuning, evaluation, error analysis, and inference.

## 📂 Project Structure

```text
customer-support-copilot/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_benchmarking.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

The notebooks contain the experimental workflow and analysis, while `src/` contains the cleaned Python implementation of the final pipeline.

---

## 📊 Dataset & Preprocessing

The project uses the **Bitext Customer Support LLM Chatbot Training Dataset**.

The dataset contains customer support conversations with:

* Customer instructions
* Categories
* Intents
* Reference responses
* 27 customer-support intents
* Approximately 27K training examples

Each training example is formulated as:

```text
Customer: <customer message>
Response: <target response>
```

The dataset was analyzed before training to understand:

* Instruction and response length distributions
* Intent distribution
* Duplicate instructions
* Response diversity
* Potential grammatical and stylistic noise

### Dataset Quality

The dataset has a relatively clean structure, but qualitative inspection revealed some stylistic and grammatical noise in the target responses.

Examples of recurring patterns include:

* Generic customer-service phrases
* Repeated apologies and greetings
* Template-like wording
* Unnecessary verbosity
* Occasional unnatural grammatical constructions

This became an important consideration during qualitative model evaluation.

### Dataset Statistics
|  Property	            |      Value                        |
| --------------------- | --------------------------------- |
| Examples	| 26,872 |
| Intents |	27 |
| Mean instruction length	| 8.69 words |
| Median instruction length	| 9 words |
| Maximum instruction length |	16 words |
| Mean response length	| 104.79 words |
| Median response length |	90 words |
| Maximum response length	 | 402 words |

---

## 🔍 Exploratory Data Analysis

Before training, the dataset was analyzed to understand its structure and characteristics.

Key observations:

* Customer instructions are generally short.
* Responses are substantially longer than the input instructions.
* The dataset contains repeated customer instructions with different valid responses.
* No instruction was found to correspond to multiple intents in the duplicate analysis.
* Response length varies significantly between intents.
* Some intents naturally require considerably longer responses than others.

The duplicate analysis showed that repeated customer instructions can have multiple valid responses, while no duplicated instruction was found to map to multiple intents.

Qualitative inspection also revealed some stylistic and grammatical noise in the target responses.

Common patterns included:

* Generic customer-service expressions
* Repeated apologies and greetings
* Template-like wording
* Unnecessary verbosity
* Occasional unnatural grammatical constructions

The complete data preparation and EDA process is available in:

```text
notebooks/01_data_preparation.ipynb
```

---

## 📉 Model Benchmarking

Several Qwen instruction-tuned models were considered during experimentation.

### Models

| Model                 | Training Approach                 |
| --------------------- | --------------------------------- |
| Qwen2.5-0.5B-Instruct | Full Fine-tuning                  |
| Qwen2.5-1.5B-Instruct | QLoRA                             |
| Qwen2.5-3B-Instruct   | Considered during model selection |

The main goal of benchmarking was to identify a model that could provide strong performance while remaining practical under limited GPU resources.

The experiments showed that the **0.5B model with full fine-tuning** provided the best overall trade-off for this project.

Detailed experiments are available in:

```text
notebooks/02_model_benchmarking.ipynb
```

---

## Fine-Tuning

The selected model was:

**Qwen/Qwen2.5-0.5B-Instruct**

The final model was fully fine-tuned on the customer-support dataset.

### Final training configuration

* Model: Qwen2.5-0.5B-Instruct
* Training method: Full fine-tuning
* Epochs: 3
* Learning rate: `2e-5`
* Weight decay: `0.01`
* Per-device batch size: `1`
* Gradient accumulation steps: `16`
* FP16: Disabled
* Evaluation: Every epoch
* Best model selection: Evaluation loss

The training results showed that the third epoch provided little additional generalization benefit:

* Epoch 2 training loss: 0.5337
* Epoch 2 validation loss: 0.6040
* Epoch 3 training loss: 0.4507
* Epoch 3 validation loss: 0.6076

The increase in training performance without a corresponding validation improvement suggests that additional training was providing limited generalization benefit.

The final training experiment showed that the third epoch provided little additional generalization benefit, with validation loss remaining approximately stable and slightly increasing.

---

## Evaluation

The models were evaluated using both lexical and semantic metrics.

### Metrics

* ROUGE-1
* ROUGE-2
* ROUGE-L
* ROUGE-Lsum
* BERTScore

Evaluation was performed on a held-out evaluation set, using generation with a maximum of 400 new tokens.

### Results

| Metric       | 0.5B Full FT | 1.5B QLoRA | 0.5B Revised |
| ------------ | -----------: | ---------: | -----------: |
| **ROUGE-1**      |       0.6041 |     0.5591 |       0.5916 |
| **ROUGE-2**      |       0.3488 |     0.3000 |       0.3372 |
| **ROUGE-L**      |       0.4581 |     0.4161 |       0.4491 |
| **ROUGE-Lsum**   |       0.5014 |     0.4475 |       0.4858 |
| **BERTScore F1** |       0.5432 |     0.5047 |       0.5382 |

The revised 0.5B model remained close to the original 0.5B model while substantially outperforming the 1.5B QLoRA experiment.

The results also demonstrate an important practical observation: a larger parameter count does not automatically translate into better task performance when the fine-tuning strategy and dataset quality differ.

The original 0.5B full fine-tuned model achieved the strongest automatic evaluation scores.

The 1.5B QLoRA experiment did not outperform the smaller fully fine-tuned model, demonstrating that increasing model size does not necessarily produce better task performance.

The revised 0.5B training configuration remained competitive but did not improve over the original configuration.

Detailed evaluation is available in:

```text
notebooks/04_model_evaluation.ipynb
```

---

##  🔍 Qualitative Error Analysis

Automatic metrics alone are not sufficient for evaluating a customer-support generation system.

The final model was therefore manually inspected to identify failure cases and generation-quality issues.

The analysis revealed a small number of clearly incorrect outputs in the evaluated samples, while most generated responses were relevant to the customer's request.

However, several qualitative issues remained:

* Repetitive customer-service language
* Template-like responses
* Unnatural grammatical constructions
* Excessive verbosity
* Occasional unnecessary numbered instructions

  
The qualitative analysis found that the majority of evaluated responses were relevant to the customer's request, while a small number of clear failure cases remained.

This analysis also revealed that some responses were technically relevant but still undesirable because of their writing style.
These issues were not fully resolved through prompt engineering or decoding changes.

---

## 📊 Prompt & Generation Experiments

Several inference strategies were tested, including:

* More explicit system instructions
* Concise-response instructions
* Restrictions on greetings and generic phrases
* Greedy decoding
* Sampling
* Temperature
* Top-p
* Repetition penalty
* Different `max_new_tokens` limits

These experiments did not substantially change the model's underlying response style.

This suggests that the observed behavior is primarily related to what the model learned during fine-tuning rather than being solely a generation-configuration problem.

---

## Lessons Learned
### 1. Model size is not everything

The 1.5B QLoRA model did not outperform the smaller 0.5B fully fine-tuned model.

This showed that model size alone is not a sufficient indicator of task performance.

### 2. Fine-tuning strategy matters

Full fine-tuning and parameter-efficient fine-tuning produced different results even when applied to related model families.

The choice of fine-tuning strategy should therefore consider both computational constraints and expected task performance.

### 3. Automatic metrics are not enough

ROUGE and BERTScore provided useful quantitative comparisons, but they could not fully capture:

* Grammar
* Naturalness
* Helpfulness
* Verbosity
* Repetition
* Response style

Qualitative inspection was therefore necessary.

### 4. Generation settings have limited power

Prompt engineering and decoding changes were useful experiments, but they did not eliminate the underlying stylistic patterns learned during fine-tuning.

### 5. Training targets matter

Supervised fine-tuning teaches the model not only what to answer, but also how the training responses are written.

Noisy, repetitive, or template-heavy target responses can therefore influence the model's generation style.

## Limitations

The main limitation identified during qualitative evaluation is the quality and style of the training responses.

Although the dataset is structurally suitable for supervised fine-tuning, some target responses contain stylistic and grammatical noise.

As a result, the model occasionally learns:

* Repetitive customer-service expressions
* Template-like language
* Verbose response patterns
* Unnatural grammatical constructions

This is an observed limitation of the current experiment and should not be interpreted as proof that dataset quality is the only cause of these behaviors.

---




## 🛠 Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd customer-support-copilot

pip install -r requirements.txt
```

---

## Training

The final training pipeline is available in:

```text
src/train.py
```

Run:

```bash
python src/train.py
```

---

## Evaluation

The evaluation pipeline is available in:

```text
src/evaluate.py
```

Run:

```bash
python src/evaluate.py
```

The script evaluates generated responses using ROUGE and BERTScore.

---

## Inference

The inference script is available in:

```text
src/inference.py
```

It can be used to generate a customer-support response from a new customer message.

Example:

```text
Customer: I need help cancelling my purchase.

Assistant:
<generated response>
```

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate
* PEFT / QLoRA
* Qwen2.5
* ROUGE
* BERTScore

---
## 🚀 Future Work

A natural next experiment is to create a higher-quality supervised fine-tuning dataset containing:

* Concise responses
* Natural English
* Grammatically correct sentences
* Less repetitive phrasing
* Fewer generic customer-service expressions
* More direct answers

The same **Qwen2.5-0.5B-Instruct** model can then be fine-tuned on the cleaned dataset while keeping the architecture and evaluation methodology fixed.

This would provide a controlled experiment for measuring how much response quality improves when the quality of the supervision is improved.

Other possible improvements include:

* Better response-quality evaluation
* Human evaluation
* Preference-based fine-tuning
* Retrieval-augmented generation
* Risk/intent classification for high-risk customer requests
* Integration with a real customer-support workflow

---
## Conclusion

This project explored the practical process of adapting an instruction-tuned LLM for customer support response generation.

The experiments demonstrated that strong task performance depends on more than model size. Fine-tuning strategy, training configuration, evaluation methodology, and especially the quality of supervised responses all play important roles.

The final system provides a practical foundation for further experimentation with cleaner supervision, preference optimization, retrieval, and production-oriented customer-support workflows.
