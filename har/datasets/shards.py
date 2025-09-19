from __future__ import annotations
import numpy as np, glob
from torch.utils.data import Dataset
from har.transforms.normalize import NormStats

class NPZShardsDataset(Dataset):
    def __init__(self, shards_glob: str, split: str, stats: NormStats | None = None):
        self.files = sorted(glob.glob(shards_glob))
        self.index = []  # list of (file_idx, sample_idx)
        self._map = []  # cache of per-file indices
        self.stats = stats

        for i, f in enumerate(self.files):
            z = np.load(f, allow_pickle=False)
            n = z["X"].shape[0]
            if "split" in z.files:
                if split == "all":
                    idx = np.arange(n)
                else:
                    m = np.array(z["split"]) == split
                    idx = np.nonzero(m)[0]
            else:
                # if shards are not split-tagged, include all
                idx = np.arange(n)
            self._map.append((f, idx))
            for j in idx:
                self.index.append((i, j))

    def __len__(self): return len(self.index)

    def __getitem__(self, k):
        file_i, sample_j = self.index[k]
        f, idx = self._map[file_i]
        z = np.load(f, allow_pickle=False)
        x = z["X"][sample_j].astype(np.float32)   # (C,T)
        y = int(z["y"][sample_j])
        if self.stats is not None:
            x = self.stats.apply(x)
        return x, y
    
    def get_dataset_statistics(self) -> dict:
        """
        Extract comprehensive dataset statistics for NPZ shards dataset.
        
        Returns:
            dict: Dictionary containing:
                - total_samples: Total number of samples
                - num_subjects: Number of unique subjects (if available)
                - num_activities: Number of unique activities
                - activities_per_subject: Dictionary mapping subject_id to list of activities
                - samples_per_subject: Dictionary mapping subject_id to sample count
                - samples_per_activity: Dictionary mapping activity_id to sample count
                - subject_activity_matrix: Dictionary showing activity distribution per subject
        """
        # Load all data to compute statistics
        all_data = []
        for i in range(len(self)):
            x, y = self[i]
            # Get subject_id from the shard if available
            file_idx, sample_idx = self.index[i]
            file_path = self._map[file_idx][0]
            z = np.load(file_path, allow_pickle=False)
            if "subject_id" in z.files:
                subject_id = z["subject_id"][sample_idx]
            else:
                subject_id = f"file_{file_idx}"
            all_data.append({"subject_id": subject_id, "activity": y})
        
        # Convert to DataFrame for easier analysis
        import pandas as pd
        df = pd.DataFrame(all_data)
        
        # Calculate statistics
        stats = {
            "total_samples": len(df),
            "num_subjects": df["subject_id"].nunique(),
            "num_activities": df["activity"].nunique(),
            "unique_subjects": sorted(df["subject_id"].unique().tolist()),
            "unique_activities": sorted(df["activity"].unique().tolist()),
            "activities_per_subject": {},
            "samples_per_subject": {},
            "samples_per_activity": {},
            "subject_activity_matrix": {}
        }
        
        # Activities per subject
        for subject in df["subject_id"].unique():
            subject_data = df[df["subject_id"] == subject]
            activities = sorted(subject_data["activity"].unique().tolist())
            stats["activities_per_subject"][subject] = activities
            stats["samples_per_subject"][subject] = len(subject_data)
        
        # Samples per activity
        for activity in df["activity"].unique():
            activity_data = df[df["activity"] == activity]
            stats["samples_per_activity"][activity] = len(activity_data)
        
        # Subject-activity matrix
        for subject in df["subject_id"].unique():
            subject_data = df[df["subject_id"] == subject]
            activity_counts = subject_data["activity"].value_counts().to_dict()
            stats["subject_activity_matrix"][subject] = activity_counts
        
        return stats