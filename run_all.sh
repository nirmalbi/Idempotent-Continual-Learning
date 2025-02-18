#! /bin/bash

# training with best hyperparameters known (from previous experiments)
for N_TASKS in  10
do
    for SEED in 0 1 2 3 4
    do 
        # no best args
        # python main.py --model="ewc_on" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --run_name="ewc_on seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        # no best args
        # python main.py --model="si" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --run_name="ewc_on seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="icarl" --load_best_args --dataset="seq-cifar100" --device="cuda:5" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="icarl seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"
        python main.py --model="der" --load_best_args --dataset="seq-cifar100" --device="cuda:5" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="der seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"
        python main.py --model="er" --load_best_args --dataset="seq-cifar100" --device="cuda:5" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="er seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"
        python main.py --model="lucir" --load_best_args --dataset="seq-cifar100" --device="cuda:5" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="lucir seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"
        python main.py --model="er_ace" --load_best_args --dataset="seq-cifar100" --device="cuda:5" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="er_ace seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"
        # no best args
        # python main.py --model="agem" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --run_name="agem seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="bic" --load_best_args --dataset="seq-cifar100" --device="cuda:5"  --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="bic seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"
        #python main.py --model="er_ace" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="er_ace seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        #python main.py --model="gdumb" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="gdumb seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        # no best args
        # python main.py --model="mer" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="mer seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        # no best args
        python main.py --model="joint" --load_best_args --dataset="seq-cifar100" --device="cuda:5" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="joint seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"
        python main.py --model="gdumb" --load_best_args --dataset="seq-cifar100" --device="cuda:5" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="gdumb seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"
        # python main.py --model="mir" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="mir seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="derpp" --load_best_args --dataset="seq-cifar100" --device="cuda:5" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="derpp seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"
        python main.py --model="xder" --load_best_args --dataset="seq-cifar100" --device="cuda:5" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="xder seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100_baseline"

    done
done


