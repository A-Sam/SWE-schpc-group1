#!/usr/bin/bash

# ---------------------------------------------------------------------------- #
#                             Script bash Settings                             #
# ---------------------------------------------------------------------------- #

cecho(){
    red='\033[0;31m'
    RED='\033[1;31m'
    grn='\033[0;32m'
    GRN='\033[1;32m'
    ylw='\033[0;33m'
    blu='\033[0;34m'
    NC='\033[0m' # No Color

    NC="\033[0m" # No Color

    echo -e "${!1}${2} ${NC}" # <-- bash
}

# ----------------------------------- Help ----------------------------------- #

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    cecho ylw ""
    cecho ylw "This script is to generate and run STREAM benchmark on all L1d, L2, L3, Memory to measure their bandwidths"
    cecho ylw ""
    cecho ylw "To run this script:"
    # cecho ylw "\$ bash build_and_run_stream_benchmarks.sh <single_or_omp> <omp_places> <omp_proc_bind> <no_of_threads> <no_of_cores> <no_of_processors>"
    cecho ylw "\$ bash build_and_run_stream_benchmarks.sh <single_or_omp> <omp_places> <omp_proc_bind> <no_of_threads>"
    cecho ylw "Example for single core STREAM benchmark on all levels:"
    cecho ylw "\$ bash build_and_run_stream_benchmarks.sh single"
    cecho ylw ""
    cecho ylw "Example for OpenMP SM STREAM benchmark on all levels:"
    cecho ylw "\$ bash build_and_run_stream_benchmarks.sh threads close 28"
    cecho ylw ""
    cecho ylw ""
    cecho RED "IMPORTANT NOTE"
    cecho red "It's user's responsibility to configure the slurm job correctly otherwise the job will fail!"
    exit;
fi

# ------------------------------------- - ------------------------------------ #

cecho ylw "-----------------------------------------------------------------------"
cecho ylw "|           SCHPC LAB - [Assignment 2] OpenMP - Group 1 (Q2)          |"
cecho ylw "|   Auto script to measure all bandwidths for l1c, l2c, l3c, memory   |"
cecho ylw "-----------------------------------------------------------------------"

# ---------------------------------------------------------------------------- #
#                     Configuration and Parameters Setting                     #
# ---------------------------------------------------------------------------- #
## Available Compute Nodes
# PARTITION           AVAIL  TIMELIMIT  NODES  STATE NODELIST
# mpp3_inter*            up    2:00:00      1  alloc mpp3r03c05s03
# mpp3_inter*            up    2:00:00      2   idle mpp3r03c05s[01-02]
# teramem_inter          up 10-00:00:0      1    mix teramem1
# cm2_inter              up    2:00:00      1  alloc i22r07c05s11
# cm2_inter              up    2:00:00     11   idle i22r07c05s[01-10,12]
# cm2_inter_large_mem    up 4-00:00:00      5  alloc i22r07c01s[01-05]
# cm4_inter_large_mem    up 4-00:00:00      9   idle hpdar03c01s[01-09]
# ---------------------------------- General --------------------------------- #
compilers=("icc") #"icc" "gcc"

declare -A compilers_flags
compilers_flags[icc]="-O3 -xCORE-AVX2 -ffreestanding -qopenmp"
compilers_flags[gcc]="-O3 -fopenmp"

timeid=$(date +T%H%M%S)

bandwidth_level_ids=("l1c" "l2c" "l3c" "mem") #"l1c" "l2c" "l3c" "mem"

declare -A bandwidth_level_info
bandwidth_level_info[l1c]="L1d Cache"
bandwidth_level_info[l2c]="L2 Cache"
bandwidth_level_info[l3c]="L3 Cache"
bandwidth_level_info[mem]="Memory"
bandwidth_level_info[mem2]="Memory2"

# ------------------------------- Architecture ------------------------------- #

#! NOTE all array sizes are multipled by 4 as instructed in https://www.cs.virginia.edu/stream/ref.html

# -------------------------------- mpp3_inter -------------------------------- #
#### L1D=32KB, L2=256KB, L3=0, L4(MemorySideCache)=4096MB
#### Total caches size = 32*1e3 + 256*1e3 + 4096*1e6 = 4096288 KB ~ 4.1 GB
# target_hostname=mpp3r03c05s

# --------------------------------- cm2_inter -------------------------------- #
#### L1D=32KB, L2=256KB, L3=18MB
#### L1d Caches 32    *1e3 -> fits 32    *1e3 / 8 = 4   k element -> to exceed L1d STREAM_ARRAY_SIZE = 4    * 4 = 16k element
#### L2  Caches 256   *1e3 -> fits 256   *1e3 / 8 = 32  k element -> to exceed L2  STREAM_ARRAY_SIZE = 32   * 4 + 16k = 128k + 16k = 144k element
#### L3  Caches 17920 *1e3 -> fits 17920 *1e3 / 8 = 2.24m element -> to exceed L3  STREAM_ARRAY_SIZE = 2.24 * 4 + 16k + 128k = 8.96m + 16k + 128k = 9.104m element
#### Total caches size = 32*1e3 + 256*1e3 + 17920*1e3 = 17920 KB ~ 18.21 MB
# available: 4 nodes (0-3)
# node 0 cpus: 0 1 2 3 4 5 6 28 29 30 31 32 33 34
# node 0 size: 15360 MB
# node 0 free: 9449 MB
# node 1 cpus: 7 8 9 10 11 12 13 35 36 37 38 39 40 41
# node 1 size: 16125 MB
# node 1 free: 12564 MB
# node 2 cpus: 14 15 16 17 18 19 20 42 43 44 45 46 47 48
# node 2 size: 16095 MB
# node 2 free: 14831 MB
# node 3 cpus: 21 22 23 24 25 26 27 49 50 51 52 53 54 55
# node 3 size: 16124 MB
# node 3 free: 11842 MB
# node distances:
# node   0   1   2   3 
#   0:  10  11  21  21 
#   1:  11  10  21  21 
#   2:  21  21  10  11 
#   3:  21  21  11  10 

# ------------------------------- HW Structure ------------------------------- #
#! For more information check: assignment_2/i22r07c05s04_topology.svg
# i22r07c05s has dual-processor nodes. Each processors he died over 1 socket -> 2 sockets / node
# Threads:              2 Threads/Core(CPU),               14 Threads/NUMA node,    28 Threads/socket,     56 Threads/node
# Core(CPU):            7 cores/NUMA node                  14/socket(processor)     28 core/node
# Socket:               2 NUMA nodes/socket(processor)     2 sockets(processors)/node 
# Node:                 1

# ------------------- possible affinities on a single node ------------------- #

# ----------------------------- Threads Affinity ----------------------------- #
# 1  threads (1  core  )
# 2  threads (1  core  )
# 14 threads (7  cores )
# 28 threads (14 cores )
# 56 threads (28 cores )
# ------------------------------ Cores Affinity ------------------------------ #
# 1  cores (1  sockets )
# 7  cores (1  sockets )
# 14 cores (1  sockets )
# 28 cores (2  sockets )
# ----------------------------- Sockets Affinity ----------------------------- #
# 1  sockets (1  node )
# 2  sockets (1  node )
# ------------------------------ Nodes Affinity ------------------------------ #
# NONE --> Assignment 3 - MPI

# ------------------------------------- - ------------------------------------ #
target_partition=cm2_inter
target_hostname=i22r07c05s
declare -A stream_array_sizes
stream_array_sizes[l1c]=2000        # Bandwidth of L1d < 4k
stream_array_sizes[l2c]=16000       # Bandwidth of L2  < 32k
stream_array_sizes[l3c]=144000      # Bandwidth of L3  < 2.24m
stream_array_sizes[mem]=9104000     # Bandwidth of Memory 9m+
stream_array_sizes[mem2]=12000000   # Bandwidth of Memory 12m+
ntimes=20

if [[ "$1" == "single" ]]; then
    cecho ylw "Single Core Mode:"
    omp_places_policy=false
    omp_process_bind_policy=false
    omp_or_single_mode=single

    no_threads=1
    no_cores=1
    no_processors=1
    no_nodes=1
else
    cecho ylw "Multi-Core Mode with OpenMP:"
    # ---------- Thread Affinity Controlling via Thread Binding/Pinning ---------- #

    # $ export OMP_PLACES={sockets[n], cores[n], threads[n],        <lowerbound>             : <length> : <stride>}
    #                     {         ...                      {<starting_point>:<length>}     : <length> : <stride>} -> starting index, go up to length
    #                     {         ...                      {<starting_point>,next cpu,...} : <length> : <stride>} -> starting index, mention places in each index
    #                     {         ...                      {core, core, core, core, ...etc                      } -> mention places directly
    omp_places_policy=$1
    numbers_re='^[0-9]+$'
    string_re='^[a-zA-Z]+$'
    if ! [[ $omp_places_policy =~ $string_re ]] ; then
        omp_places_policy_id=$(echo "$omp_places_policy" | sed "s/\:/_/g")
        omp_places_policy_id=$(echo "$omp_places_policy_id" | sed "s/{\|}//g")
    else
        omp_places_policy_id=$omp_places_policy
    fi

    # $ export OMP_PROC_BIND={true, false, master(threads level), close(OMP_PLACES level), spread(OMP_PLACES level)}
    omp_process_bind_policy=$2

    no_threads=$3
    no_cores=$(expr  $(expr $3 + 2 - 1) / 2 ) # to get ceiling
    no_cores=$(( $no_cores > 1 ? $no_cores : 1 ))
    no_processors=$(expr  $(expr $no_cores + 14 - 1) / 14 ) # to get ceiling
    no_processors=$(( $no_processors > 1 ? $no_processors : 1 ))
    no_nodes=1
    cecho blu "Config: ${no_threads} threads, ${no_cores} cores/cpus, ${no_processors} processors, ${no_nodes} nodes"

    omp_or_single_mode=omp_${omp_places_policy_id}_${omp_process_bind_policy}
fi

# ---------------------------- cm4_inter_large_mem --------------------------- #
#### L1D=48KB, L2=1280KB, L3=60MB
#### Total caches size = 48*1e3 + 1280*1e3 + 60*1e6 = 61328 KB ~ 61.3 MB
# target_hostname=hpdar03c01s

# ---------------------------------------------------------------------------- #
#                                   Execution                                  #
# ---------------------------------------------------------------------------- #

pushd STREAM

# ---------------- Create a Makefil and A job per cache level ---------------- #

# Iterate over jobs for multiple runs to exclude odd anomalies
for compiler in "${compilers[@]}"; do
    for level_id in "${bandwidth_level_ids[@]}"; do
        cecho ylw "-----------------------------------------------------------------"
        cecho ylw "| Starting a new benchmark for : ${bandwidth_level_info[$level_id]}"

        uid=run${timeid}_${compiler}_${level_id}_${target_hostname}_${stream_array_sizes[$level_id]}_${omp_or_single_mode}_${no_threads}t_${no_cores}c_${no_processors}p_${no_nodes}n
        mkdir ${uid}
        pushd ${uid}

        # ---------------------------- Create the Makefile --------------------------- #
        if [[ $2 == "noft" || $4 == "noft" ]]; then # no/first touch logic
            cecho ylw "First-Touch is ${RED} DEACTIVATED"
            popd
            mv ${uid} ${uid}_noft 
            uid=${uid}_noft
            pushd ${uid}
cat > Makefile << anchor
all: stream

clean:
	rm -f ${uid}_* *.o

stream: ../stream_noft.c
	${compiler} ${compilers_flags[$compiler]} -DSTREAM_ARRAY_SIZE=${stream_array_sizes[$level_id]} -DNTIMES=${ntimes} ../stream_noft.c -o stream_${uid}

remake : clean all
anchor
        else
            cecho ylw "First-Touch is ${GRN} ACTIVATED(default)"
cat > Makefile << anchor
all: stream

clean:
	rm -f ${uid}_* *.o

stream: ../stream.c
	${compiler} ${compilers_flags[$compiler]} -DSTREAM_ARRAY_SIZE=${stream_array_sizes[$level_id]} -DNTIMES=${ntimes} ../stream.c -o stream_${uid}

remake : clean all
anchor
        fi

        make all -j8

        cecho ylw "| Makefile for ${uid} is created.."

        # ----------------- Create the job script to submit to SLURM ----------------- #
cat > job_${uid}.sh << anchor
#!/bin/bash

# SLURM
#SBATCH -J stream_${uid}
#SBATCH -o output_${uid}.log
#SBATCH -e error_${uid}.log
#!SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --time=00:03:00
#SBATCH --verbose
#SBATCH --mail-type=end
#SBATCH --mail-user=${EMAIL}
#SBATCH --partition=${target_partition}
#SBATCH --nodes=${no_nodes}
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#!SBATCH --threads-per-core=2
#!SBATCH --cores-per-socket=7
#!SBATCH --cpus-per-task=${no_processors}
#!SBATCH --sockets-per-node=2

export SLURM_CPU_BIND=verbose

# OMP
# TODO fix and use newer KMP_AFFINITY controller!!
unset KMP_AFFINITY
export OMP_NUM_THREADS=${no_threads}
export OMP_PLACES=${omp_places_policy}
export OMP_PROC_BIND=${omp_process_bind_policy}
export OMP_DISPLAY_AFFINITY=true
export OMP_DISPLAY_ENV=true

scontrol -dd show job \$SLURM_JOB_ID > scontrol.out

srun -v ./stream_${uid}
anchor

        cecho ylw "| Job script for ${uid} is created.."

        sbatch -v job_${uid}.sh
        
        cecho ylw "| Benchmark ended for : ${bandwidth_level_info[$level_id]}"
        cecho ylw "-----------------------------------------------------------------"
        
        # TODO fix this hardcoded value
        sleep 2

        popd
    done
done


cecho ylw "----------------- Submitted jobs : squeue -u $USER -----------------"
cecho ylw "| $(squeue -u $USER)"
cecho ylw "----------------------------------------------------------------------"

popd

# ------------------------------------- - ------------------------------------ #