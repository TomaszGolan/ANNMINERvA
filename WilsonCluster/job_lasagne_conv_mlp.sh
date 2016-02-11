#!/bin/bash
#PBS -S /bin/bash
#PBS -N lasagne-conv-mnv
#PBS -j oe
#PBS -o ./lasagne_conv_out_job.txt
#PBS -l nodes=1:gpu,walltime=24:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email #PBS -m n
# OUTFILENAME="./lasagne_conv_out_job`date +%s`.txt"

# print identifying info for this job
echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

# these are broken?...
# nCores=$['cat ${PBS_COREFILE} | wc --lines']
# nNodes=$['cat ${PBS_NODEFILE} | wc --lines']
# echo "NODEFILE nNodes=$nNodes (nCores=$nCores):"

cat ${PBS_NODEFILE}

cd $HOME
source python_bake_lasagne.sh

cd ${PBS_O_WORKDIR}
echo "PBS_O_WORKDIR is `pwd`"
GIT_VERSION=`git describe --abbrev=12 --dirty --always`
echo "Git repo version is $GIT_VERSION"

# Always use fcp to stage any large input files from the cluster file server
# to your job's control worker node. All worker nodes have attached 
# disk storage in /scratch.

# There is no fcp on the gpu nodes...
# /usr/local/bin/fcp -c /usr/bin/rcp tevnfsp:/home/perdue/Datasets/mnist.pkl.gz /scratch
# ls /scratch

cp /home/perdue/ANNMINERvA/Lasagne/lasagne_triamese_minerva.py ${PBS_O_WORKDIR}

export THEANO_FLAGS=device=gpu,floatX=float32
python lasagne_triamese_minerva.py -n 200 -t -r 0.001 -d "/phihome/perdue/theano/data/skim_data_convnet_target0.pkl.gz"

# Always use fcp to copy any large result files you want to keep back
# to the file server before exiting your script. The /scratch area on the
# workers is wiped clean between jobs.

# not really large, but okay... but, no fcp available
# /usr/local/bin/fcp -c /usr/bin/rcp mlp_best_model.pkl /home/perdue
# the pkl should just be in my launch dir...

exit 0