import wandb

device = None
wandb_project = "graph-senn"
wandb_entity = "jonas-juerss"
def init(args):
    if args.use_wandb:
        wandb_args = dict(
            project=wandb_project,
            entity=wandb_entity,
            config=args
        )
        if args.wandb_name is not None:
            wandb_args["name"] = args.wandb_name
        wandb.init(**wandb_args)
        return wandb.config
    return args


def log(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
