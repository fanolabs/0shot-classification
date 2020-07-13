# argument order: dataset, caption, gpu

if [ "$1" == "SMP18" ]
then
	dim=12
else
	dim=12
fi

conda activate intentTorch110
cd ~/project/spin_outlier_ws/code
export CUDA_VISIBLE_DEVICES=$3

python main_spin_outlier.py -data $1 -fd $dim -cap $2 

python ~/general/sdmail.py -to 19074431r@connect.polyu.hk -msg "$1 $2 zero shot fin"
source ~/.bashrc
