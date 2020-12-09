def print_metrics(metrics):
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"NDCG: {metrics['ndcg']:.4f}")


def print_times(pipeline, num_users):
    timers = list(pipeline.timers.items())
    timers.sort(key=lambda x: -x[1])

    for k, v in timers:
        print(f"{k}: {v:.2f}s ({(v * 1000 / num_users):.2f} ms/user)")
