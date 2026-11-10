---
title: DzSentiment
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Sentiment for Darija, Arabic, French and English comments
---

Paste customer comments in Algerian Darija, Arabic, French or English and get negative, neutral or positive for each one.

Model: [okba-bouziane/dzsentiment-xlmr-base](https://huggingface.co/okba-bouziane/dzsentiment-xlmr-base), XLM-RoBERTa base fine-tuned on 45,516 labelled comments and tweets (test mean macro-F1 0.704 across the four languages). Runs on CPU; comments are not stored.
