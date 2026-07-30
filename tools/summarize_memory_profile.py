#!/usr/bin/env python3
import argparse
import json


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="memory_profile/torch_memory_events.jsonl")
    args = parser.parse_args()

    events = []
    with open(args.path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "allocated_mib" in record:
                events.append(record)

    if not events:
        raise SystemExit("No CUDA memory events found")

    print(
        "stage | allocated | reserved | peak allocated | driver used | "
        "free | inactive split | RSS (all MiB)"
    )
    for event in events:
        print(
            f"{event['stage']} | {fmt(event.get('allocated_mib'))} | "
            f"{fmt(event.get('reserved_mib'))} | "
            f"{fmt(event.get('max_allocated_mib'))} | "
            f"{fmt(event.get('driver_used_mib'))} | "
            f"{fmt(event.get('cuda_free_mib'))} | "
            f"{fmt(event.get('inactive_split_mib'))} | "
            f"{fmt(event.get('process_rss_mib'))}"
        )

    peak = max(events, key=lambda event: event.get("max_allocated_mib", 0.0))
    least_free = min(events, key=lambda event: event.get("cuda_free_mib", float("inf")))
    most_fragmented = max(
        events, key=lambda event: event.get("inactive_split_mib", 0.0)
    )
    print("\nKey findings")
    print(
        f"Peak observed after stage '{peak['stage']}': "
        f"{fmt(peak.get('max_allocated_mib'))} MiB allocated"
    )
    print(
        f"Lowest free memory at '{least_free['stage']}': "
        f"{fmt(least_free.get('cuda_free_mib'))} MiB"
    )
    print(
        f"Largest inactive split at '{most_fragmented['stage']}': "
        f"{fmt(most_fragmented.get('inactive_split_mib'))} MiB"
    )

    if any(event["stage"].startswith("oom_after_") for event in events):
        oom = next(
            event for event in reversed(events)
            if event["stage"].startswith("oom_after_")
        )
        print(
            f"OOM occurred after '{oom.get('previous_stage')}', with "
            f"allocated={fmt(oom.get('allocated_mib'))} MiB, "
            f"reserved={fmt(oom.get('reserved_mib'))} MiB, "
            f"free={fmt(oom.get('cuda_free_mib'))} MiB."
        )


if __name__ == "__main__":
    main()
