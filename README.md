# ALKDRec
This repository contains the source code for **Active Large Language Model-based Knowledge Distillation for Session-based Recommendation (ALKDRec)**. 

If you would like to review the full paper with its appendix, please refer to `full_paper_with_appendix.pdf` included in this repository.

---

## Quick Start （Finally Step）
To run the **ALKDRec** model with different backbones, use the following Jupyter Notebooks:

- **FPMC backbone**: `distillation-Atten-FPMC.ipynb`
- **STAMP backbone**: `distillation-Atten-STAMP.ipynb`
- **Atten-Mixer backbone**: `distillation-Atten-Mixer.ipynb`

> **Note**: Cell 1 in each notebook includes options for the dataset and ablation studies.
>
> **Ablation Study**: To assess the effectiveness of ALKDRec's module design, we compare it with several variants:
>
> - **(1) TR**: Distills knowledge from the **T**eacher **R**ecommender by mimicking its predictions, which fine-tunes the well-trained student recommender for KD.
> - **(2) random**: Distills knowledge from LLM by randomly sampling a subset of instances.
> - **(3) hardest/easiest**: Distills knowledge from LLM for the hardest/easiest instances based on their difficulties.
> - **(4) RAD-BC**: Replaces our active learning strategy in ALKDRec with the **r**obust **a**ctive **d**istillation on **b**inary **c**lassification tasks, which models only correct and incorrect instances. The number of correct instances in RAD-BC is set to match the total number of effective/similar instances in ALKDRec.

---

## Workflow
The following steps outline the process for training and running ALKDRec:

### Step 1: Train Teacher and Student Models
Train the teacher and student models separately using the following notebooks:
- `teacher.ipynb` (Train the teacher model)
- `student.ipynb` (Train the student model)

### Step 2: Organize Predictions
Organize the predictions from the teacher and student models for further processing:
- `teacher_and_student_prediction.ipynb`

### Step 3: Generate Hints Using LLM
Leverage the large language model (LLM) to summarize the teacher's predictions and generate hints for knowledge distillation:
- `LLM_learn_from_teacher.ipynb`

### Step 4: Active Learning
Perform active learning to refine the knowledge distillation process:
- `active learning-Games.ipynb` 
- `active learning-ML.ipynb`
### NOTICE
**The outputs from all the above steps are already included in this package for your convenience.**

---
