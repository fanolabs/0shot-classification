# argument order: dataset, caption, seed, gpu

if [ "$1" == "SMP18" ]
then
	dim=12
else
	dim=12
fi

conda activate intentTorch110
cd ~/project/spin_outlier_ws/code
export CUDA_VISIBLE_DEVICES=$4

# python main_outlier.py -data $1 -fd $dim -cap $2 -s $3 -p 0.25
# python main_outlier.py -data $1 -fd $dim -cap $2 -s $3 -p 0.5
python main_outlier.py -data $1 -fd $dim -cap $2 -s $3 -p 0.75
