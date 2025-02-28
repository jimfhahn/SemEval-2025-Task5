# Jim-at-SemEval-2025-Task-5
The [SemEval-2025 Task 5](https://github.com/jd-coderepos/llms4subjects/) calls for the utilization of LLM capabilities to apply controlled subject labels to record descriptions in the multilingual library collection of the [German National Library of Science and Technology](https://en.wikipedia.org/wiki/German_National_Library_of_Science_and_Technology). 

## System Overview
The multilingual BERT ensemble system described herein produces [GND subject labels](https://en.wikipedia.org/wiki/Integrated_Authority_File) for various record types, including articles, books, conference papers, reports, and theses. Input a title and abstract in German or English to generate GND subject labels.

## Train
The [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced/) software package was used to train BERT models for GND classification based on examples from the TIB "All Subjects" dataset. A curated set of that data spilt into validation and train is available from [Hugging Face](https://huggingface.co/datasets/jimfhahn/SemEval2025-Task5-Curated-Data)

## Test
This code was developed to test which set of models contributed to the highest scores using 1000 rows of held out data as the gold standard.

## Inference
Inference code generates labels and aggregates label confidence scores so the BERT models work as an ensemble during inference.

## GitHub CoPilot Attribution
Jim was assisted by [GitHub Copilot](https://copilot.github.com/), for development of the inference and testing code.
