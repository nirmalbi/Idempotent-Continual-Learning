for N_TASKS in  10
do
    for SEED in 0 1 2 3 4
    do
    #python main.py --model="er" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="er seed $SEED $N_TASKS tasks" --experiment_name="er/cifar100/buffer500"
    python main.py --model="ider" --load_best_args --savecheckpoint=True  --class_balance=True  --dataset="seq-cifar100" --device="cuda:7" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name=" er+id seed $SEED $N_TASKS tasks" --experiment_name="idempotent/cifar100/buffer500"
    #python main.py --model="ider" --load_best_args --savecheckpoint=True  --class_balance=True  --dataset="seq-cifar100" --device="cuda:7" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=2000 --run_name=" er+id  seed $SEED $N_TASKS tasks" --experiment_name="idempotent/cifar100/buffer2000"
    done
done