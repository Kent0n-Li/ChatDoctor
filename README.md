<p align="center" width="80%">
<img src="fig/logo.png" style="width: 40%; min-width: 300px; display: block; margin: auto;">
</p>


# ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/HUANGLIZI/ChatDoctor/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Page](https://img.shields.io/badge/Web-Page-yellow)](https://www.python.org/downloads/release/python-390/) 

## Overview
ChatDoctor is a next-generation AI doctor model that is based on the [LLaMA](https://github.com/facebookresearch/llama) model. The goal of this project is to provide patients with an intelligent and reliable healthcare companion that can answer their medical queries and provide them with personalized medical advice.

The ChatDoctor is an advanced language model that is specifically designed for medical applications. It has been trained on a large corpus of medical literature and has a deep understanding of medical terminology, procedures, and diagnoses. This model serves as the foundation for ChatDoctor, enabling it to analyze patients' symptoms and medical history, provide accurate diagnoses, and suggest appropriate treatment options.

The ChatDoctor model is designed to simulate a conversation between a doctor and a patient, using natural language processing (NLP) and machine learning techniques. Patients can interact with the ChatDoctor model through a chat interface, asking questions about their health, symptoms, or medical conditions. The model will then analyze the input and provide a response that is tailored to the patient's unique situation.

One of the key features of the ChatDoctor model is its ability to learn and adapt over time. As more patients interact with the model, it will continue to refine its responses and improve its accuracy. This means that patients can expect to receive increasingly personalized and accurate medical advice over time.



## Examples:

Below are some results of the our model. 
<p align="center" width="100%">
<img src="fig/chat_example1.png" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>
 

## Abstract
Recent large language models (LLMs) in the general domain, such as ChatGPT, have shown remarkable success in following instructions and producing human-like responses. However, such language models have not been learned individually and carefully for the medical domain, resulting in poor diagnostic accuracy and inability to give correct recommendations for medical tests, medications, etc. 
We collected more than 700 diseases and their corresponding symptoms, recommended medications, and required medical tests, and then generated 5K doctor-patient conversations. By fine-tuning models of doctor-patient conversations, these models emerge with great potential to understand patients' needs, provide informed advice, and offer valuable assistance in a variety of medical-related fields. The integration of these advanced language models into healthcare can revolutionize the way healthcare professionals and patients communicate, ultimately improving the overall quality of care and patient outcomes. In addition, we will open source all code and datasets and model weights to advance the further development of dialogue models in the medical field.

## Introduction
The development of instruction-following large language models (LLMs) such as ChatGPT has garnered significant attention due to their remarkable success in instruction understanding and human-like response generation.
These auto-regressive LLMs are pre-trained over web-scale natural languages by predicting the next token and then fine-tuned to follow large-scale human instructions.
Also, they have shown strong performances over a wide range of NLP tasks and generalizations to unseen tasks, demonstrating their potential as a unified solution for various problems such as natural language understanding, text generation, and conversational AI.
However, the exploration of such general-domain LLMs in the medical field remains relatively untapped, despite the immense potential they hold for transforming healthcare communication and decision-making.
The specific reason is that the existing models do not learn the medical field in detail, resulting in the models often giving wrong diagnoses and wrong medical advice when playing the role of a doctor. By fine-tuning the large language dialogue model on the data of doctor-patient conversations, the application of the model in the medical field can be significantly improved. Especially in areas where medical resources are scarce, ChatDoctor can be used for initial diagnosis and triage of patients, significantly improving the operational efficiency of existing hospitals.

Since large language models such as ChatGPT are in a non-open source state, we used Meta's LLaMA and first trained a generic conversation model using 52K instruction-following data provided by Stanford Alpaca, and then fine-tuned the model on our collected physician-patient conversation dataset.
The main contributions of our method are three-fold:
1) We designed a process framework for fine-tuning large language models in the medical domain.
2) We collected a training data with 5,000 doctor-patient conversations for fine-tuning the large language model.
3) We validate that the fine-tuned bigrams with medical domain knowledge have real potential for clinical application.


## Physician and patient conversation dataset</h2>
The first step in building a physician-patient conversation dataset is to collect the disease database that serves as the gold standard. Therefore, we collected and organized a database of diseases, which contains about 700 diseases with their relative symptoms, medical tests, and recommended medications. To train high-quality conversation models on an academic budget, we input each message from the disease database separately as a prompt into the ChatGPT API to automatically generate instruction data. It is worth noting that our prompts to the ChatGPT API contain the gold standard of diseases and symptoms, and drugs, so our fine-tuned ChatDoctor is not only able to achieve ChatGPT's conversational fluency but also higher diagnostic accuracy compared to ChatGPT. We finally collected 5K doctor-patient conversation instructions and named it InstructorDoctor-5K.

## Training of the model
We build ChatDoctor utilizing Meta's LLaMA model, a distinguished publicly accessible LLM.
Notably, in spite of its 7 billion parameters, LLaMA has been reported that LLaMA's efficacy can attain competitive or superior outcomes in comparison to the considerably larger GPT-3 (with 175 billion parameters) on several NLP benchmarks.
LLaMA's performance improvement was achieved by amplifying the magnitude of training data, as opposed to parameter quantity.
Specifically, LLaMA was trained on 1.4 trillion tokens, procured from publicly accessible data repositories such as CommonCrawl and arXiv documents.
We utilize conversation demonstrations synthesized via ChatGPT and subsequently validated by medical practitioners to fine-tune the LLaMA model, in accordance with the Stanford Alpaca training methodology.
The fine-tuning process was conducted using 6 A*100 GPUs for a duration of 30 minutes.
The hyperparameters employed in the training process were as follows: the total batch size of 192, a learning rate of 2e-5, a total of 3 epochs, a maximum sequence length of 512 tokens, a warmup ratio of 0.03, with no weight decay.




 ## Setup:
 In a conda env with pytorch available, run:
```
pip install -r requirements.txt
```

 ## Interactive Demo Page:
We are developing the Demo Page and it is coming soon!

 ## Data and model:
 ### 1. ChatDoctor Training Dataset:
You can download the following training dataset
InstructorDoctor-5K: [link](https://drive.google.com/file/d/1nDTKZ3wZbZWTkFMBkxlamrzbNz0frugg/view?usp=sharing)
 
 ### 2. Model Weights:
In order to download the checkpoints, fill this form: [link](https://forms.office.com/Pages/ResponsePage.aspx?id=lYZBnaxxMUy1ssGWyOw8ij06Cb8qnDJKvu2bVpV1-ANUMDIzWlU0QTUxN0YySFROQk9HMVU0N0xJNC4u).
Place the model weights file in the ./pretrained folder.

 ## How to fine-tuning

 ```python
torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --data_path ./chatdoctor5k.json \
    --bf16 True \
    --output_dir pretrained \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
    --tf32 True
 ```
 
 ## How to inference
 You can build a ChatDoctor model on your own machine and communicate with it.
 ```python
python chat.py
 ```

## Limitations
We emphasize that ChatDoctor is for academic research only and any commercial use and clinical use is prohibited. There are three factors in this decision: First, ChatDoctor is based on LLaMA and has a non-commercial license, so we necessarily inherited this decision. Second, our model is not licensed for healthcare-related purposes. Also, we have not designed sufficient security measures, and the current model still does not guarantee the full correctness of medical diagnoses.




## Reference

ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge

```
@misc{ChatDoctor,
  title={ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge},
  author={Li, Yunxiang and Li, Zihan and Zhang, Kai and Dan, Ruilong and Dohopolski, Michael and Zhang, You},
  publisher = {GitHub},
  journal = {GitHub repository},
  year={2023}
}
```
<p align="center" width="100%">
<img src="fig/chat_example3.png" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

<p align="center" width="100%">
<img src="fig/chat_example4.png" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>