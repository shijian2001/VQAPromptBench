vqa_models=("llavav1.5-7b-trl" "llavav1.5-7b-10k-tune" "llavav1.5-7b-259k-tune")

for vqa_model in "${vqa_models[@]}"; do
    python eval_bench.py --vqa_model "$vqa_model"
    wait
done

echo "Finshed"
