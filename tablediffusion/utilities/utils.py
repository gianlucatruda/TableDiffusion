import os
import random
from pathlib import Path

import git
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from torch import nn
from tqdm import tqdm
from utilities import load_and_prep_data


def weights_init(m):
    # TODO docstring
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def gather_object_params(obj, prefix="", clip=249):
    # TODO docstring
    return {
        prefix + k: str(v)[:clip] if len(str(v)) > clip else v for k, v in obj.__dict__.items()
    }


def set_random_seed(seed=None):
    """Sets randomisation seed for math, np, torch."""

    # Generate random seed if none specified
    if seed is None:
        seed = np.random.randint(10000)

    # Apply the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_synthesisers(
    datasets,
    synthesisers,
    exp_name,
    exp_id,
    datadir,
    repeats=3,
    repodir="./",
    epsilon_values=[1.0],
    generate_fakes=True,
    fake_sample_path=None,
    fake_data_path=None,
    with_benchmark=False,
    ctgan_epochs=30,
    cuda=True,
    metaseed=42,
):
    # TODO docstring

    datadir = Path(datadir)
    if not os.path.exists(datadir):
        raise RuntimeError(f"{datadir} does not exist")

    # Make any directories that will be needed if they don't exist
    if fake_data_path is not None and not os.path.exists(fake_data_path):
        os.makedirs(fake_data_path)
    if fake_sample_path is not None and not os.path.exists(fake_sample_path):
        os.makedirs(fake_sample_path)

    if cuda:
        cuda = bool(torch.cuda.is_available())
        print(f"CUDA status: {cuda}")

    # Generate random seeds for each repeat from `metaseed`
    np.random.seed(metaseed)
    _seeds = np.random.randint(10000, size=repeats)

    repo = git.Repo(repodir)
    githash = str(repo.head.object.hexsha)
    gitmessage = str(repo.head.object.message)
    gitbranch = str(repo.active_branch)
    git_remote_url = str(repo.remotes.origin.url)

    client = MlflowClient()
    client.set_experiment_tag(
        exp_id, "mlflow.note.content", f"{gitbranch}:{githash[:6]}:{gitmessage}"
    )

    for repeat in tqdm(range(1, repeats + 1), desc="Repeats", leave=True, colour="red"):

        # Determine random seed for the repeat (from metaseed)
        _seed = _seeds[repeat - 1]
        set_random_seed(_seed)

        for dataset, data_params in datasets.items():

            # Load and transform dataset (privately?)
            path = datadir / data_params["path"]
            X, _X, processor = load_and_prep_data(dataset=dataset, datadir=datadir, verbose=False)
            disc_cols = [c for c, dtype in data_params["data_types"] if "categorical" in dtype]
            print(f"Loaded {dataset} dataset {X.shape} from {path}")

            if with_benchmark and fake_data_path is not None:
                print("Benchmarking with CTGAN...")
                from ctgan.synthesizers import CTGAN as CTGANSynthesizer

                ctgan = CTGANSynthesizer(epochs=ctgan_epochs, cuda=cuda)
                ctgan.fit(train_data=X, discrete_columns=disc_cols)
                X_fake_benchmark = ctgan.sample(X.shape[0])
                print("Saving fake data...")
                pd.DataFrame(X_fake_benchmark).to_csv(
                    Path(fake_data_path) / f"fake_{dataset}_CTGAN_Synthesiser_0.0_{repeat}.csv",
                    index=False,
                )

            for eps in tqdm(epsilon_values, desc="Epsilon", leave=True, colour="green"):

                for _synth, (synth, init_params, fit_params, extra_params) in tqdm(
                    synthesisers.items(), desc="Synths"
                ):

                    raw_data = extra_params.get("use_raw_data", False)

                    run_name = f"{exp_name}_{dataset}_{_synth}_{eps}_{repeat}"

                    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):

                        mlflow.set_tags(
                            {
                                "mlflow.source.git.commit": githash,
                                "mlflow.source.git.branch": gitbranch,
                                "mlflow.source.git.repoURL": git_remote_url,
                                "mlflow.note.content": f"{gitbranch}:{githash[:6]}:{gitmessage}",
                                "synthesiser": _synth,
                                "repeat": f"{repeat}/{repeats}",
                                "random_seed.run": _seed,
                                "random_seed.meta": metaseed,
                                "random_seed.seeds": _seeds,
                            }
                        )

                        mlflow.log_params(
                            {
                                "gpu_properties": str(torch.cuda.get_device_properties(0))
                                if cuda
                                else "cpu",
                                "dataset.dataset_name": str(dataset),
                                "dataset.shape_raw": X.shape,
                                "dataset.shape_transformed": _X.shape,
                                "dataset.drop_cols": list(data_params["drop_cols"]),
                                "dataset.disc_cols": list(disc_cols),
                                "init_params": init_params,
                                "fit_params": fit_params,
                                "extra_params": extra_params,
                            }
                        )

                        print(f"Training {_synth} on {dataset} with epsilon={eps}...")

                        try:
                            # Re-initialise the model and customise budget
                            model = synth(cuda=cuda, epsilon_target=eps, **init_params)

                            # Fit the synth on the dataset
                            if raw_data:
                                model = model.fit(
                                    X.copy(), epsilon=eps, discrete_columns=disc_cols, **fit_params
                                )
                            else:
                                model = model.fit(_X.copy(), epsilon=eps, **fit_params)

                            mlflow.log_metrics(
                                {"epsilon": model._eps, "elapsed_batches": model._elapsed_batches},
                                step=model._elapsed_batches,
                            )

                            if generate_fakes:
                                # Generate fake dataset
                                print("Generating fake data...")
                                X_fake = pd.DataFrame(model.sample(_X.shape[0]))

                                if not raw_data:
                                    # Reverse the transformation
                                    X_fake = pd.DataFrame(
                                        processor.inverse_transform(X_fake.values),
                                        columns=X.columns,
                                    )

                                # Log samples and stats for fake data to MLflow
                                pd.set_option("display.max_columns", 200)
                                mlflow.log_text(
                                    str(X_fake.describe(include="all")),
                                    f"fake_stats_{dataset}_{_synth}_{eps}.txt",
                                )
                                pd.set_option("display.max_columns", None)
                                if fake_sample_path is not None:
                                    _sample_path = f"{fake_sample_path}/fake_sample_{dataset}_{_synth}_{eps}_{repeat}.txt"
                                    X_fake.sample(30).to_csv(_sample_path, index=False)
                                    mlflow.log_artifact(_sample_path, "samples")

                                if fake_data_path is not None:
                                    # Save fake dataset
                                    print("Saving fake data...")
                                    X_fake.to_csv(
                                        Path(fake_data_path)
                                        / f"fake_{dataset}_{_synth}_{eps}_{repeat}.csv",
                                        index=False,
                                    )

                        except Exception as e:
                            mlflow.set_tag("error", f"{e}\t{e.args}")
                            raise e
