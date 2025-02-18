for N_TASKS in  5
do
    for SEED in 43
    do
    #python main.py --model="der" --load_best_args --dataset="seq-tinyimg" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="der seed $SEED $N_TASKS tasks" --experiment_name="seq-tinyimg"
    #python main.py --model="derid" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="derpp seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
    python main.py --model="derloss" --load_best_args --dataset="seq-cifar10" --device="cuda:6" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="derlossplot seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar10"
    done
done