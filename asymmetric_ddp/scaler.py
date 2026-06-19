def scale_loss_for_asymmetric_ddp(loss, local_batch_size, global_batch_size, world_size):
    """
    The Mathematical Core of Asymmetric DDP:
    Standard PyTorch DDP averages gradients across all GPUs by dividing by `world_size`.
    If Node 1 processes 90 images and Node 2 processes 10 images, DDP will blindly
    average them 50/50, causing Node 2's tiny batch to overpower Node 1's massive batch.
    
    By scaling the local loss before backward() using this exact formula, we mathematically
    trick DDP's internal NCCL reduction engine into computing the true, globally-weighted 
    mean gradient without needing to write custom C++ synchronization hooks!
    """
    scaling_factor = (local_batch_size * world_size) / global_batch_size
    return loss * scaling_factor
