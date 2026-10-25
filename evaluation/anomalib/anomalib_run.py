# evaluation/anomalib/anomalib_run.py: anomalib CLI entry point that runs on local datasets and backbone weights

# local_setup must be imported first: it sets environment variables before anomalib is imported.
from local_setup import setup_local

from lightning.pytorch import seed_everything

from anomalib.cli.cli import AnomalibCLI
from anomalib.loggers import configure_logger


class LocalAnomalibCLI(AnomalibCLI):
    """AnomalibCLI that applies seed_everything, which the stock CLI declares but never calls."""

    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        seed = self.config[self.subcommand].get("seed_everything")
        # The CLI default is True, which is not a seed.
        if isinstance(seed, int) and not isinstance(seed, bool):
            seed_everything(seed, workers=True)


if __name__ == "__main__":
    setup_local()
    configure_logger()
    LocalAnomalibCLI()
