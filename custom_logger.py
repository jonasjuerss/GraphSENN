import wandb

device = None
def init(args):
    if args["use_wandb"]:
        wandb.init(project="graph-senn", entity="jonas-juerss", config=args)
        return wandb.config
    return args


def log(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
