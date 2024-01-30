#!/bin/bash

#SBATCH --mail-user=svshirke@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J Homography
#SBATCH --output=/home/svshirke/CV/HW1/Phase2/Code/Logs/Homography%j.out
#SBATCH --error=/home/svshirke/CV/HW1/Phase2/Code/Logs/Homography%j.err

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1


#SBATCH -p academic
#SBATCH -t 23:59:00
echo "Starting myscript"
python3 Code/Train_sup.py --BasePath /home/svshirke/CV/HW1/Phase2/Data --CheckPointPath /home/svshirke/CV/HW1/Phase2/Code/model/ --ModelType Unsup --NumEpochs 100 --DivTrain 1 --MiniBatchSize 64