for N_TASKS in  5
do
    for SEED in 0 1 2 3 4
    do
 
    #python main.py --model="er" --load_best_args --savecheckpoint=True --dataset="seq-cifar10" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="er seed $SEED $N_TASKS tasks" --experiment_name="try"
    python main.py --model="ider" --load_best_args --savecheckpoint=True  --class_balance=True  --dataset="seq-cifar10" --device="cuda:6" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=200 --run_name=" er+id seed $SEED $N_TASKS tasks" --experiment_name="idempotent/cifar10/buffer200"
    python main.py --model="ider" --load_best_args --savecheckpoint=True  --class_balance=True  --dataset="seq-cifar10" --device="cuda:6" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name=" er+id seed $SEED $N_TASKS tasks" --experiment_name="idempotent/cifar10/buffer500"

    done
done