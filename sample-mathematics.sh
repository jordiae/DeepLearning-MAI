#!/usr/bin/env bash

# Extract the question, answer pairs from the train-easy subset such that answers are either True or False
# egrep -r . -e "True|False" -c
# ./numbers__is_factor_composed.txt:666666
# ./comparison__pair_composed.txt:363904
# ./numbers__is_prime_composed.txt:666666
# ./numbers__is_prime.txt:666666
# ./numbers__is_factor.txt:666666

egrep -r data/mathematics/mathematics_dataset-v1.0/train-easy -h -e "True|False" -B 1 > data/mathematics/mathematics_dataset-v1.0/train_easy_true_false.txt
sed -i '/--/d' data/mathematics/mathematics_dataset-v1.0/train_easy_true_false.txt

# Concat even and odd lines (ie. answers and questions)
sed '$!N;s/\n/ /' data/mathematics/mathematics_dataset-v1.0/train_easy_true_false.txt > data/mathematics/mathematics_dataset-v1.0/train_easy_true_false_concat.txt

# For resource constraints, we are going to subsample the data

seed=1

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

shuf  --random-source=<(get_seeded_random $seed) -o data/mathematics/mathematics_dataset-v1.0/train_easy_true_false_concat_subsampled.txt < data/mathematics/mathematics_dataset-v1.0/train_easy_true_false_concat.txt
head  data/mathematics/mathematics_dataset-v1.0/train_easy_true_false_concat_subsampled.txt -n 120000 | uniq > tmp.txt
head -n 100000 tmp.txt > data/mathematics/mathematics_dataset-v1.0/train_easy_true_false_concat_subsampled.txt
rm tmp.txt
rm data/mathematics/mathematics_dataset-v1.0/train_easy_true_false.txt
rm data/mathematics/mathematics_dataset-v1.0/train_easy_true_false_concat.txt