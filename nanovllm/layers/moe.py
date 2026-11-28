"""Eager expert parallelism on the same ranks as tensor-parallel attention."""
import torch
import torch.distributed as dist


def exchange_rows(rows, send_counts, recv_counts):
    received = rows.new_empty((sum(recv_counts), *rows.shape[1:]))
    dist.all_to_all_single(received, rows.contiguous(), recv_counts, send_counts)
    return received


def expert_parallel_forward(hidden_states, router, experts, first_expert, local_count):
    rank = dist.get_rank()
    world = dist.get_world_size()
    # TP Attention gives each rank identical rows. Exactly one source routes each row.
    local_input = hidden_states[rank::world]
    _, scores, indices = router(local_input)
    flat_experts = indices.flatten()
    destinations = flat_experts // local_count
    order = destinations.argsort(stable=True)
    token_rows = order // indices.size(1)

    send = destinations.bincount(minlength=world)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    send_counts, recv_counts = send.tolist(), recv.tolist()

    received = exchange_rows(local_input[token_rows], send_counts, recv_counts)
    expert_ids = exchange_rows(flat_experts[order], send_counts, recv_counts)
    computed = torch.empty_like(received)
    for expert_id in range(first_expert, first_expert + local_count):
        rows = torch.where(expert_ids == expert_id)[0]
        if rows.numel():
            computed[rows] = experts[str(expert_id)](received[rows])

    # Reverse the splits: returned[i] belongs to the i-th packed route at this source.
    returned = exchange_rows(computed, recv_counts, send_counts)
    weighted = returned * scores.flatten()[order, None]
    output = torch.zeros_like(hidden_states)
    output.index_add_(0, rank + world * token_rows, weighted)
    dist.all_reduce(output)
    return output
