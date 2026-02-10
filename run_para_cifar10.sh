for N_TASKS in  5
do
    for SEED in 0 1 2 3 4
    do
        for A in 0.05
        do
            for B in 0.05
            do
                for C in 0.5
                do
                    for D in 0
                    do
                        for E in 1
                        do
                            for F in 0
                            do
 
    python main.py --model="er" --load_best_args --savecheckpoint=True --dataset="seq-cifar10" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="er seed $SEED $N_TASKS tasks" --experiment_name="try"
    python main.py --model="idempotent2" --load_best_args --savecheckpoint=True --weighta=$A   --weightb=$B --weightc=$C --class_balance=True --alpha_bfp=$D --weightema=$F --weightmask=$E --dataset="seq-cifar10" --device="cuda:6" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=200 --run_name=" er+id $A der $B er $C bfp $D mask $E bfp_distill $F seed $SEED $N_TASKS tasks" --experiment_name="idempotent/cifar10/buffer200"
    python main.py --model="idempotent2" --load_best_args --savecheckpoint=True --weighta=$A   --weightb=$B --weightc=$C --class_balance=True --alpha_bfp=$D --weightema=$F --weightmask=$E --dataset="seq-cifar10" --device="cuda:6" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name=" er+id $A der $B er $C bfp $D mask $E bfp_distill $F seed $SEED $N_TASKS tasks" --experiment_name="idempotent/cifar10/buffer500"

                            done
                        done
                    done
                done
            done
        done
    done
done