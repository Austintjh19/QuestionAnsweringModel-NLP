# A Span-Based Question Answering Model

## Overview

This is a novel span-based Question Answering (QA) model capable of addressing both answerable and unanswerable questions with remarkable accuracy and reliability. The model mimics the human reading process, incorporating a tri-phased strategy: 

1. **Sketchy Reading Phase:** Performs an initial evaluation of answer likelihood to identify unanswerable questions.
2. **Intensive Reading Phase:** Predicts the start and end positions of the answer span for answerable questions.
3. **Verification Phase:** Assesses the credibility of the predicted span, adding an additional layer of validation.

This multi-phase approach significantly improves the model’s ability to handle unanswerable questions, a challenge often overlooked by existing QA systems.

---

## User Guide

### Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

### Training the Model

The model was trained on **H100-96 GPU** on the **SoC cluster** with the following training times:

- **Squad-v1.1:** ~60 minutes  
- **Squad-v2.0:** ~90 minutes  

#### Pre-configured Hyperparameters:
- **Epochs:** 2  
- **Learning Rate (lr):** 2e-5  

**Note:**  
- Models trained on **Squad-v2.0** can handle both answerable and unanswerable questions.  
- The best hyperparameters are already configured.

#### Command for Training:

**Squad-V1.1**  
```bash
python model.py --train --save_path "retro_reader_model.pth" --epochs 2 --lr 2e-5 --dataset "squad"
```

**Squad-V2.0**  
```bash
python model.py --train --save_path "retro_reader_model.pth" --epochs 1 --lr 2e-5 --dataset "squad_v2"
```

---

### Testing the Model

Testing generates a `predictions.json` file, which can be evaluated directly using the evaluation script.

#### Command for Testing:

**Squad-V1.1**  
```bash
python model.py --test --model_path "retro_reader_model.pth" --dataset "squad"
```

**Squad-V2.0**  
```bash
python model.py --test --model_path "retro_reader_model.pth" --dataset "squad_v2"
```

---

### Making Inferences

The model predicts an answer for a given question and context.

#### Command for Inference:

**Squad-V1.1**  
```bash
python model.py --inference --question "What is the capital of France?" --context "Paris is the capital of France, known for its architecture including the Eiffel Tower." --model_path "retro_reader_model.pth" --dataset "squad"
```

**Squad-V2.0**  
```bash
python model.py --inference --question "What is the capital of France?" --context "Paris is the capital of France, known for its architecture including the Eiffel Tower." --model_path "retro_reader_model.pth" --dataset "squad_v2"
```

---

### Evaluating the Model

After generating `predictions.json` via testing, evaluate the results using the provided evaluation script.

#### Command for Evaluation:

**Squad-V1.1**  
```bash
python evaluate-v2.0.py dev-v1.1.json predictions.json
```

**Squad-V2.0**  
```bash
python evaluate-v2.0.py dev-v2.0.json predictions.json
```