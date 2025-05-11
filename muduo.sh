#!/bin/bash
#SBATCH -J muduo_test        
#SBATCH -p kshdtest           
#SBATCH -N 1                  
#SBATCH --gres=dcu:1         
#SBATCH --ntasks-per-node=1   
#SBATCH -o muduo_job.out      
#SBATCH -e muduo_job.err      
#SBATCH --time=01:00:00       


module load compiler/dtk/24.04.2


cd /public/home/xdzs2025_cubic/muduo_gpu  


./muduo data/stories110M.bin data/tokenizer.bin data/input_prompt.txt