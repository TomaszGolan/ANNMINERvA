#!/bin/bash
#PBS -S /bin/bash
#PBS -N lasagne-conv-mnv
#PBS -j oe
#PBS -o ./lasagne_conv_out_job.txt
# not 2 #PBS -l nodes=gpu2:gpu:ppn=1,walltime=24:00:00
#PBS -l nodes=gpu1:gpu:ppn=1,walltime=24:00:00
# #PBS -l nodes=1:gpu,walltime=24:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email #PBS -m n

NEPOCHS=16
NEPOCHS=6
LRATE=0.001
L2REG=0.0001

DATAFILENAME="/phihome/perdue/theano/data/minosmatch_nukecczdefs_127x50_xuv_me1Bmc.hdf5"
SAVEMODELNAME="./lminervatriamese_betaprime`date +%s`.npz"
PYTHONPROG="minerva_triamese_betaprime.py"

# TODO: try passing this in 
# SAVEMODELNAME="./lminervatriamese_beta1457480019_beta_v1r0.npz"
# START_FROM="-p -s $SAVEMODELNAME"
# START_FROM=""

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
DIRTY=`echo $GIT_VERSION | perl -ne 'print if /dirty/'`
if [[ $DIRTY != "" ]]; then
  echo "Git repo contains uncomitted changes! Please commit your changes"
  echo "before submitting a job. If you feel your changes are experimental,"
  echo "just use a feature branch."
  echo ""
  echo "Changed files:"
  git diff --name-only
  echo ""
  # exit 0
fi

# Always use fcp to stage any large input files from the cluster file server
# to your job's control worker node. All worker nodes have attached 
# disk storage in /scratch.

# There is no fcp on the gpu nodes...
# /usr/local/bin/fcp -c /usr/bin/rcp tevnfsp:/home/perdue/Datasets/mnist.pkl.gz /scratch
# ls /scratch

cp /home/perdue/ANNMINERvA/Lasagne/${PYTHONPROG} ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/Lasagne/minerva_ann_*.py ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/Lasagne/network_repr.py ${PBS_O_WORKDIR}
cp /home/perdue/ANNMINERvA/Lasagne/predictiondb.py ${PBS_O_WORKDIR}

cat << EOF
python ${PYTHONPROG} -l \
  -n $NEPOCHS \
  -r $LRATE \
  -g $L2REG \
  -s $SAVEMODELNAME \
  -d $DATAFILENAME \
# --imgh 68
# $START_FROM
EOF
export THEANO_FLAGS=device=gpu,floatX=float32
python ${PYTHONPROG} -l \
  -n $NEPOCHS \
  -r $LRATE \
  -g $L2REG \
  -s $SAVEMODELNAME \
  -d $DATAFILENAME 
# --imgh 68
# $START_FROM

# nepochs and lrate don't matter for prediction, but setting them for log-file
# homogeneity
# cat << EOF
# python ${PYTHONPROG} -t \
#   -n $NEPOCHS \
#   -r $LRATE \
#   -g $L2REG \
#   -d $DATAFILENAME \
#   -s $SAVEMODELNAME -a
# EOF
# python ${PYTHONPROG} -t \
#   -n $NEPOCHS \
#   -r $LRATE \
#   -g $L2REG \
#   -d $DATAFILENAME \
#   -s $SAVEMODELNAME -a

# Always use fcp to copy any large result files you want to keep back
# to the file server before exiting your script. The /scratch area on the
# workers is wiped clean between jobs.

# not really large, but okay... but, no fcp available
# /usr/local/bin/fcp -c /usr/bin/rcp mlp_best_model.pkl /home/perdue
# the pkl should just be in my launch dir...

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} finished "`date`" jobid ${PBS_JOBID}"
exit 0
