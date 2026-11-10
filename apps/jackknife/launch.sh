#!/bin/bash
SIMU_NPROC=1                     # Number of simulation processes
DASK_NB_WORKERS=5                # Number of Dask workers
DASK_NB_THREAD_PER_WORKER=1      # Number of threads per Dask workers

SCHEFILE=scheduler.json

export DEISA_DASK_SCHEDULER_ADDRESS="tcp://localhost:8786"

# Launch Dask Scheduler in a 1 Node and save the connection information in $SCHEFILE
echo launching Scheduler
mpirun -np 1 dask scheduler \
    --interface lo \
    --scheduler-file=$SCHEFILE \
    --protocol tcp \
    --port '8786' &
dask_sch_pid=$!

# Wait for the SCHEFILE to be created 
while ! [ -f $SCHEFILE ]; do
    sleep 1
    echo -n .
done

echo Scheduler booted, launching workers
mpirun -np 1 dask worker \
    --interface lo \
    --nworkers ${DASK_NB_WORKERS} \
    --nthreads ${DASK_NB_THREAD_PER_WORKER} \
    --scheduler-file=${SCHEFILE} &  
dask_worker_pid=$!

sleep 1

# Launch the analytics
echo Running analytics
mpirun -np 1 python deisaclient.py imager.yml ../imager/ingest.config &
analytics_pid=$!

sleep 1

# Launch the simulation code
echo Running Simulation 
mpirun -np 6 python ../imager/imager.py imager.yml ../imager/ingest.config
sim_pid=$!

sleep 1

kill $dask_worker_pid

wait $dask_worker_pid $analytics_pid $sim_pid

kill $dask_sch_pid