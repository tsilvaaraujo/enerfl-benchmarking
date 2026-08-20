# EnerFL Benchmarking

Reproducible experiments for measuring the performance, communication overhead, resource usage, energy consumption, and carbon emissions of federated learning workloads with EnerFL. The repository runs Flower workloads on HPC clusters using Singularity and supports Slurm and OAR schedulers.

## Getting the code

The profiler lives in a separate repo (`enerfl`) and is vendored here as a git
submodule at `enerfl/`. Clone with submodules:

```bash
git clone --recurse-submodules <benchmarking-url>
git submodule update --init
```

Bump the pinned profiler revision:

```bash
git -C enerfl pull origin main
git add enerfl && git commit -m "Bump enerfl"
```

## Run with Apptainer or Singularity on HPC

The experiments use SIF containers and work with either
[Apptainer](https://apptainer.org/) or Singularity. Both runtimes work without
a root daemon and are commonly available as environment modules on HPC
clusters. The scheduler backend is pluggable: pick Slurm or OAR (Grid'5000)
via `--scheduler` or `SCHEDULER` in `scripts/conf/cluster.conf`.

### Build the SIF images

Load the container runtime provided by the cluster, if it is exposed through
environment modules:

```bash
module load apptainer
```

From the project root, build the server and client images with Apptainer:

```bash
apptainer build --fakeroot singularity/superlink.sif singularity/superlink.def
apptainer build --fakeroot singularity/supernode.sif singularity/supernode.def
```

With Singularity, use the equivalent commands:

```bash
singularity build --fakeroot singularity/superlink.sif singularity/superlink.def
singularity build --fakeroot singularity/supernode.sif singularity/supernode.def
```

The build requires network access because the definition files use the
`python:3.11-slim` Docker image and install Python dependencies from package
indexes. It also requires either fakeroot support or root privileges. On a
system without fakeroot, ask the cluster administrator to enable it or build
the images on another compatible Linux host and transfer the resulting SIF
files.

Verify that both images were created and can be inspected:

```bash
apptainer inspect singularity/superlink.sif
apptainer inspect singularity/supernode.sif
```

The experiment scripts use `singularity/superlink.sif` and
`singularity/supernode.sif` by default. Override these locations with
`SUPERLINK_SIF` and `SUPERNODE_SIF` in `scripts/conf/cluster.conf`. Subsequent
runs reuse the existing images; rebuild them after changing either `.def`
file or its dependencies.

### Configure the experiment matrix

Edit `scripts/conf/experiments.conf` to change datasets, algorithms, models,
client counts, repetitions, or Slurm partition names. The `COMBOS` variable
encodes which `model=dataset[,dataset...]` pairs are valid — the job
generator only emits the intersection so `OpenLLaMA` never trains on
`CIFAR-10`, etc.

### Submit the full matrix (serial)

From `scripts/`:

```bash
bash submit_all_jobs.sh
```

This regenerates `job_list.txt` from the matrix and then submits every
combination one at a time, chaining each run's orchestrator on the previous
run's server job. This prevents a large matrix from competing for the same
cluster resources.

For finer control, call `run_matrix_experiment.sh` directly:

```bash
bash run_matrix_experiment.sh --regen         # force regenerate job_list first
bash run_matrix_experiment.sh --from 42       # skip first 42 lines (resume)
bash run_matrix_experiment.sh --limit 10      # only submit first 10 jobs
bash run_matrix_experiment.sh --scheduler oar # pick the scheduler backend
bash run_matrix_experiment.sh --when night --walltime 10:00:00  # OAR night/weekend job
bash run_matrix_experiment.sh --when day      # OAR daytime job (<=1h, 09:00-19:00 workdays)
PROFILER_MODE=off bash run_matrix_experiment.sh
```

### Scaling clients

Each FL client runs on its own physical node — one SuperNode (one client)
per node, with no co-location. A run with `NUM_CLIENTS` clients therefore
reserves `NUM_CLIENTS + 1` nodes (1 server + N clients). All training runs
on CPU.

### Scripts layout

```
scripts/
├── submit_all_jobs.sh          # entry point — regen + serial matrix run
├── generate_job_list.sh        # writes job_list.txt from the matrix
├── run_matrix_experiment.sh    # reads job_list.txt, submits chained runs
├── run_intrusion_experiment.sh # profiler-overhead validation experiment
├── job.sh                      # per-run orchestrator (server + clients)
├── run_server.sh               # SuperLink + flwr run on the server node
├── run_client.sh               # one SuperNode on a client node
├── conf/
│   ├── cluster.conf            # scheduler, partition, walltime, container
│   └── experiments.conf        # matrix + partitions
├── lib/
│   ├── common.sh               # shared helpers
│   ├── backend_oar.sh          # OAR submit/exec backend
│   └── backend_slurm.sh        # Slurm submit/exec backend
├── job_list.txt
└── logs/
```
