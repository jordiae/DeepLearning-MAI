#!/usr/bin/env bash
wget https://console.cloud.google.com/storage/browser/_details/mathematics-dataset/mathematics_dataset-v1.0.tar.gz -P data
mkdir data/mathematics
tar -xvf data/mathematics_dataset-v1.0.tar.gz -C data/mathematics
rm data/mathematics_dataset-v1.0.tar.gz
