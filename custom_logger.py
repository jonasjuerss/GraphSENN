import wandb

device = None
wandb_project = "graph-senn"
wandb_entity = "jonas-juerss"
def init(args):
    if args.use_wandb:
        wandb.init(project=wandb_project, entity=wandb_entity, config=args)
        return wandb.config
    return args


def log(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
