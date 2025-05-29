Fine-Tuned LLaMA-3.2 Medical Chain-of-Thought
Project Overview
This project involves fine-tuning the LLaMA-3.2-3B-Instruct model (4-bit quantized) on the FreedomIntelligence/medical-o1-reasoning-SFT dataset to enhance its ability to perform medical question-answering with chain-of-thought (CoT) reasoning. The model is optimized for medical diagnosis tasks, producing structured responses with <think> and <response> tags to reflect reasoning and final answers. The fine-tuned model is evaluated using ROUGE-L scores and deployed to the Hugging Face Hub for public access.
Objectives

Fine-tune a LLaMA-3.2-3B-Instruct model for medical question-answering with chain-of-thought reasoning.
Format input data with <think> and <response> tags to align with the dataset structure.
Optimize training for memory-constrained environments (e.g., T4 GPU).
Evaluate model performance using ROUGE-L scores.
Deploy the fine-tuned model to the Hugging Face Hub.

Dataset
The dataset used is FreedomIntelligence/medical-o1-reasoning-SFT (English subset) from Hugging Face. It contains medical questions, complex chain-of-thought reasoning (Complex_CoT), and corresponding responses. The dataset is split into training and validation sets (100 samples for validation).
Methodology

Data Preprocessing:

Load the dataset and convert it to a Pandas DataFrame.
Format each sample with Question:, <think>, and <response> tags to align with the desired output structure.
Handle null or non-string values and split into training and validation sets.


Model Setup:

Use the unsloth/Llama-3.2-3B-Instruct-bnb-4bit model with 4-bit quantization for memory efficiency.
Apply LoRA (Low-Rank Adaptation) with parameters r=16, lora_alpha=32, and lora_dropout=0.05.


Training:

Fine-tune the model using the SFTTrainer from the trl library with optimized settings for T4 GPUs (e.g., per_device_train_batch_size=1, gradient_accumulation_steps=8, max_seq_length=1024).
Log training metrics to Weights & Biases (W&B) for monitoring.
Perform evaluation every 50 steps using the validation set.


Inference:

Generate responses for sample medical questions, ensuring outputs include <think> and <response> tags.
Handle incomplete responses by adjusting max_new_tokens and checking for proper tag closure.


Evaluation:

Compute ROUGE-L scores to assess the similarity between predicted and reference responses.
Use a small subset of validation data for demonstration purposes.


Deployment:

Save the fine-tuned model and tokenizer locally.
Upload to the Hugging Face Hub under abdulsamad99/medical-fine-tuning.
Create and push a model card for documentation.



Tools and Libraries

Python: Core programming language.
Unsloth: Optimized model loading and fine-tuning for memory efficiency.
Transformers & TRL: Model fine-tuning and training pipeline.
Datasets: Data loading from Hugging Face.
Pandas: Data manipulation.
PyTorch: Backend for model training and inference.
Weights & Biases (W&B): Training monitoring and logging.
ROUGE-Score: Evaluation of model performance.
Hugging Face Hub: Model storage and deployment.

Installation

Clone the repository:
git clone https://github.com/your-username/fine-tuned_llama-3.2_medical_chain_of_thought.git
cd fine-tuned_llama-3.2_medical_chain_of_thought


Install required libraries:
pip install unsloth bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
pip install rouge-score wandb


Set up Weights & Biases (W&B):

Replace the W&B API key in the script with your own:os.environ["WANDB_API_KEY"] = "your-wandb-api-key"




Set up Hugging Face Hub:

Replace the Hugging Face token in the script with your own:api = HfApi(token="your-hf-token")





Usage

Run the training script (medical_fine_tuning.py):
python medical_fine_tuning.py

This will:

Load and preprocess the dataset.
Fine-tune the LLaMA-3.2 model.
Log training progress to W&B.
Evaluate the model on the validation set.
Save and upload the model to abdulsamad99/medical-fine-tuning.


Perform inference:

The script includes an inference section to generate responses for sample medical questions.
Example output format:Question: A 56-year-old patient presents with sudden chest pain radiating to the left arm. What is the most likely diagnosis?
<think>Reasoning steps here...</think><response>Most likely diagnosis: Acute myocardial infarction</response>




Evaluate ROUGE-L scores:

The script computes ROUGE-L scores on a subset of validation data for demonstration.



Results

Training: The model was fine-tuned for 200 steps with a learning rate of 1e-5, optimized for T4 GPUs using 4-bit quantization and LoRA.
Inference: The model generates structured responses with <think> and <response> tags, suitable for medical question-answering.
Evaluation: ROUGE-L scores indicate the model’s ability to produce responses similar to the reference answers.
Deployment: The model is available at abdulsamad99/medical-fine-tuning on the Hugging Face Hub.

Visualizations

Training and validation loss curves are logged to Weights & Biases.
Sample inference outputs demonstrate the model’s reasoning and diagnosis capabilities.

Limitations

The model is fine-tuned on a specific dataset and may not generalize to all medical scenarios.
Memory constraints on T4 GPUs limit the sequence length and batch size.
The ROUGE-L evaluation is performed on a small subset; larger-scale evaluation could provide more insights.
The model may occasionally produce incomplete responses, requiring adjustments to max_new_tokens.

Future Improvements

Fine-tune on additional medical datasets to improve generalization.
Experiment with higher-capacity models (e.g., LLaMA-3.2-8B) if hardware permits.
Implement automated hyperparameter tuning for LoRA and training arguments.
Enhance evaluation with additional metrics (e.g., BLEU, human evaluation).
Add support for multilingual medical datasets.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset provided by FreedomIntelligence/medical-o1-reasoning-SFT on Hugging Face.
Model based on unsloth/Llama-3.2-3B-Instruct-bnb-4bit.
Thanks to the Unsloth, Transformers, and TRL teams for their optimized libraries.

