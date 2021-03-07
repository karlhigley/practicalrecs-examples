from practicalrecs_examples.metrics import metric_fns
import torch as th


def print_metrics(metrics):
    for metric_name in metric_fns.keys():
        if metric_name in metrics.keys():
            # print(f"{metric_name.capitalize()}: {metrics[metric_name]:.4f}")
            print(f"{metric_name.capitalize()}: {th.mean(metrics[metric_name]):.4f}")
    # print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    # print(f"NDCG: {metrics['ndcg']:.4f}")


def print_times(pipeline, num_users):
    timers = list(pipeline.timers.items())
    timers.sort(key=lambda x: -x[1])

    for k, v in timers:
        print(f"{k}: {v:.2f}s ({(v * 1000 / num_users):.2f} ms/user)")
