import torch as th


class SimulatedEvalDataset:
    def __init__(self, num_users, num_items, device):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

    def __len__(self):
        return self.num_users

    def __getitem__(self, index):
        user_id = int(index)

        return {
            "user_ids": th.tensor([user_id], device=self.device),
            "interactions": self.simulate_interactions(user_id),
        }

    def simulate_interactions(self, user_id):
        num_interactions = th.randint(1, 100, (1,)).squeeze().item()
        item_ids = th.randint(0, self.num_items, (num_interactions,))
        targets = th.empty_like(item_ids).fill_(1.0)

        return self._sparse_vector(
            user_id, item_ids, targets, self.num_users, self.num_items, self.device
        )

    def _sparse_vector(self, user_id, item_ids, targets, num_users, num_items, device):
        item_indices = item_ids.to(dtype=th.int64)
        user_indices = th.empty_like(item_indices, dtype=th.int64).fill_(user_id)
        item_labels = targets.to(dtype=th.float64)

        return th.sparse.FloatTensor(
            th.stack([user_indices, item_indices]), item_labels, (num_users, num_items)
        ).to(device=device)
