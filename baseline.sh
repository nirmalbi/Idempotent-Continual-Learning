for N_TASKS in  5
do
    for SEED in 43
    do
    python main.py --model="der" --load_best_args --dataset="seq-tinyimg" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="der seed $SEED $N_TASKS tasks" --experiment_name="seq-tinyimg"

    done
done