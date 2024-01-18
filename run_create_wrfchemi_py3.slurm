#!/bin/bash

#SBATCH --job-name=PythonEmission           # nom du job
#SBATCH --partition=zen4                    # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=4                          # nombre de taches
#SBATCH --ntasks-per-node=4                 # nombre de taches MPI par noeud
#SBATCH --mem=50GB                          # memory limit
#SBATCH --time=24:00:00                     # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=PythonEmission%j.out       # nom du fichier de sortie
#SBATCH --error=PythonEmission%j.error      # nom du fichier d'erreur (ici en commun avec la sortie)


# Link the necessary files from the SI directory


module purge
module load python/meso-3.9
conda activate my_python3

mpirun python create_wrfchemi.py3

