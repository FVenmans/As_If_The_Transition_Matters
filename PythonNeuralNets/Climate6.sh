####sequence of commands for "Optimal climate policy as if the transition matters"

##check parameters
sed -i 's/N_episodes: [0-9]*/N_episodes: 1003/' /files/config/run/Climate6.yaml
sed -i 's/RRA: [0-9]*\.[0-9]*/RRA: 1.35/' /files/config/constants/Climate6.yaml 
sed -i 's/bellman_weight: [0-9]*/bellman_weight: 1/' /files/config/constants/Climate6.yaml
sed -i 's/Sigmak: [0-9]*\.[0-9]*/Sigmak: 0.00022/' /files/config/constants/Climate6.yaml
sed -i 's/SigmaS: [0-9]*\.[0-9]*/SigmaS: 0.0007/' /files/config/constants/Climate6.yaml
#run the main code
python run_deepnet.py 

#### find the most recently created run directory #ls -td lists directories sorted by modification time, and head -1 picks the most recent one. 
RUNDIR=$(ls -td /files/runs/Climate6/*/* | head -1) # For fixed output dir :RUNDIR=/files/runs/Climate6/current_run
echo "Using run directory: $RUNDIR"

echo "post_processing v1"
sed -i 's/versionidx = [0-9]*/versionidx = 1/' post_process_climate6.py
grep 'versionidx = 1' post_process_climate6.py && echo "Change to v1 successful" || echo "WARNING: versionidx not found"
python post_process_climate6.py STARTING_POINT=LATEST hydra.run.dir=$RUNDIR 2>&1 | tee -a $RUNDIR/output.log #saves ckpt11 (checkpoint 1 is at episode2)

sed -i 's/N_episodes: [0-9]*/N_episodes: 201/' /files/config/run/Climate6.yaml # $RUNDIR/.hydra/config.yaml  if export $RUNDIR was used.
grep 'N_episodes: 201' /files/config/run/Climate6.yaml || { echo "WARNING: N_episodes not found"; exit 1; }
sed -i 's/RRA: [0-9]*\.[0-9]*/RRA: 2.0/' /files/config/constants/Climate6.yaml  
grep 'RRA: 2.0' /files/config/constants/Climate6.yaml || { echo "WARNING: RRA: 2.0  not found"; exit 1; } 
sed -i 's/bellman_weight: [0-9]*/bellman_weight: 10/' /files/config/constants/Climate6.yaml  
grep 'bellman_weight: 10' /files/config/constants/Climate6.yaml || { echo "WARNING: Bellman_weight not found"; exit 1; }
echo "Run training for RRA=2"
python run_deepnet.py hydra.run.dir=$RUNDIR  2>&1 | tee -a $RUNDIR/output.log 

sed -i 's/RRA: [0-9]*\.[0-9]*/RRA: 3.0/' /files/config/constants/Climate6.yaml
grep 'RRA: 3.0' /files/config/constants/Climate6.yaml || { echo "WARNING: RRA: 3.0  not found"; exit 1; } 
echo "Run training for RRA=3"
python run_deepnet.py hydra.run.dir=$RUNDIR 2>&1 | tee -a $RUNDIR/output.log

sed -i 's/N_minibatch_size: 200/N_minibatch_size: 1000/' /files/config/run/Climate6.yaml
grep 'N_minibatch_size: 1000' /files/config/run/Climate6.yaml || { echo "WARNING: N_minibatch_size: 1000  not found"; exit 1; } 
echo "Run training for minibatch 1000"
python run_deepnet.py hydra.run.dir=$RUNDIR 2>&1 | tee -a $RUNDIR/output.log

#read -p "Press Enter to continue..."

#RRA=4 does not converge.
#sed -i 's/RRA: [0-9]*\.[0-9]*/RRA: 4.0/' /files/config/constants/Climate6.yaml
#grep 'RRA: 4.0' /files/config/constants/Climate6.yaml || { echo "WARNING: RRA: 4.0  not found"; exit 1; } 
#sed -i 's/bellman_weight: [0-9]*/bellman_weight: 50/' /files/config/constants/Climate6.yaml  
#grep 'bellman_weight: 50' /files/config/constants/Climate6.yaml || { echo "WARNING: Bellman_weight not found";  exit 1; } 
#echo "Run training for RRA=4"
#python run_deepnet.py hydra.run.dir=$RUNDIR   2>&1 | tee -a $RUNDIR/output.log
CKPT=$(ls $RUNDIR/ckpt-*.index | sed 's/\.index//' | sort -V | tail -1) 
echo $CKPT > $RUNDIR/CKPT_full_model.txt
echo $CKPT    2>&1 | tee -a $RUNDIR/output.log
echo "post_processing v2"
sed -i 's/versionidx = [0-9]*/versionidx = 2/' post_process_climate6.py || echo "WARNING: versionidx not found"
python post_process_climate6.py STARTING_POINT=LATEST hydra.run.dir=$RUNDIR  2>&1 | tee -a $RUNDIR/output.log

echo "Set SigmaS to 0.0007 and Sigmak to 0.0"
sed -i 's/SigmaS: [0-9]*\.[0-9]*/SigmaS: 0.0007/' /files/config/constants/Climate6.yaml
grep 'SigmaS: 0.0007' /files/config/constants/Climate6.yaml || { echo "WARNING: SigmaS not found"; exit 1; }
sed -i 's/Sigmak: [0-9]*\.[0-9]*/Sigmak: 0.0/' /files/config/constants/Climate6.yaml
grep 'Sigmak: 0\.0\b' /files/config/constants/Climate6.yaml || { echo "WARNING: Sigmak not found"; exit 1; }
sed -i 's/N_minibatch_size: 1000/N_minibatch_size: 200/' /files/config/run/Climate6.yaml
echo "Run training"
python run_deepnet.py STARTING_POINT=LATEST hydra.run.dir=$RUNDIR   2>&1 | tee -a $RUNDIR/output.log 
echo "post_processing v3"
sed -i 's/versionidx = [0-9]*/versionidx = 3/' post_process_climate6.py || echo "WARNING: versionidx not found"
python post_process_climate6.py STARTING_POINT=LATEST hydra.run.dir=$RUNDIR  2>&1 | tee -a $RUNDIR/output.log

echo "Set SigmaS to 0 and Sigmak to 0.00022"
sed -i 's/SigmaS: [0-9]*\.[0-9]*/SigmaS: 0.0/' /files/config/constants/Climate6.yaml
grep 'SigmaS: 0\.0\b' /files/config/constants/Climate6.yaml || { echo "WARNING: SigmaS not found"; exit 1; }
sed -i 's/Sigmak: [0-9]*\.[0-9]*/Sigmak: 0.00022/' /files/config/constants/Climate6.yaml
grep 'Sigmak: 0.00022' /files/config/constants/Climate6.yaml || { echo "WARNING: Sigmak not found"; exit 1; }
echo "Run training"
python run_deepnet.py STARTING_POINT=$CKPT hydra.run.dir=$RUNDIR 2>&1 | tee -a $RUNDIR/output.log
echo "post_processing v4"
sed -i 's/versionidx = [0-9]*/versionidx = 4/' post_process_climate6.py || echo "WARNING: versionidx not found"
python post_process_climate6.py STARTING_POINT=LATEST hydra.run.dir=$RUNDIR  2>&1 | tee -a $RUNDIR/output.log #could go back to another ckpt

echo "Set SigmaS to 0.0 and Sigmak to 0.0"
sed -i 's/SigmaS: [0-9]*\.[0-9]*/SigmaS: 0.0/' /files/config/constants/Climate6.yaml
grep 'SigmaS: 0\.0\b' /files/config/constants/Climate6.yaml || { echo "WARNING: SigmaS not found"; exit 1; }
sed -i 's/Sigmak: [0-9]*\.[0-9]*/Sigmak: 0.0/' /files/config/constants/Climate6.yaml
grep 'Sigmak: 0\.0\b' /files/config/constants/Climate6.yaml || { echo "WARNING: Sigmak not found"; exit 1; }
echo "Run training"
python run_deepnet.py STARTING_POINT=LATEST hydra.run.dir=$RUNDIR  2>&1 | tee -a $RUNDIR/output.log
echo "post_processing v5"
sed -i 's/versionidx = [0-9]*/versionidx = 5/' post_process_climate6.py || echo "WARNING: versionidx not found"
python post_process_climate6.py STARTING_POINT=LATEST hydra.run.dir=$RUNDIR  2>&1 | tee -a $RUNDIR/output.log

python post_process_boxplots.py  hydra.run.dir=$RUNDIR 2>&1 | tee -a $RUNDIR/output.log