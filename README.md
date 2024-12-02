# AI Fine-Tuning: Message Classification and Paraphrasing with BERT and GPT-2

**AI Fine-Tuning** is a comprehensive project demonstrating how to fine-tune pre-trained language models like **BERT** and **GPT-2** for specialized tasks such as message classification and paraphrasing. This repository includes scripts for training a BERT classifier, classifying new messages, and generating paraphrased versions of messages using GPT-2.

## Table of Contents

- [Overview](#overview)
- [Dataset Preparation](#dataset-preparation)
- [Installation](#installation)
- [Scripts and Usage](#scripts-and-usage)
  - [1. Training the BERT Classifier](#1-training-the-bert-classifier)
  - [2. Classifying Messages with the Trained BERT Model](#2-classifying-messages-with-the-trained-bert-model)
  - [3. Paraphrasing Messages with GPT-2](#3-paraphrasing-messages-with-gpt-2)
- [Results](#results)
- [References](#references)
- [License](#license)

## Overview

This project showcases the fine-tuning of **BERT** and **GPT-2** models for:

- **Message Classification**: Training BERT to categorize messages as related or not related to a specific topic.
- **Paraphrasing**: Utilizing GPT-2 to generate diverse paraphrased versions of positive messages.

By following the provided scripts and instructions, you can replicate the process and adapt it to your own datasets and tasks.

## Dataset Preparation

### 1. Positive Messages (`goodmessage.csv`)

- **Description**: Contains messages relevant to the specific topic.
- **Expected Column**: `posmessage`

### 2. Negative Messages (`badmessage.csv`)

- **Description**: Contains messages not relevant to the topic.
- **Expected Column**: `badmessage`

### 3. Messages to Classify (`eldenring_data.csv`)

- **Description**: Contains messages that need to be classified.
- **Expected Column**: `messageText`

**Note**: Ensure that your CSV files are properly formatted and placed in the `c:/python39/` directory or adjust the file paths accordingly in the scripts.

## Installation

### Prerequisites

- **Python 3.6** or higher
- **Node.js** and **npm** (if applicable)
- **pip** package manager

### Install Required Libraries

Open your terminal or command prompt and run:

```bash
pip install pandas transformers torch scikit-learn datasets
