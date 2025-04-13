import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def load_flights(filename):
    flights = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            flight_id = row[0]
            start_time = int(row[1])
            end_time = int(row[2])
            coords = list(map(float, row[3:]))
            waypoints = [(coords[i], coords[i+1], coords[i+2]) for i in range(0, len(coords), 3)]
            flights.append({
                "id": flight_id,
                "start_time": start_time,
                "end_time": end_time,
                "waypoints": waypoints
            })
    return flights

def euclidean(p1, p2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

def generate_dense_waypoints(waypoints):
    dense = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i+1]
        dist = euclidean(start, end)
        steps = max(1, int(dist))
        for j in range(steps + 1):
            t = j / steps
            point = tuple(start[k] + t * (end[k] - start[k]) for k in range(3))
            dense.append(point)
    return dense

def generate_timed_dense_waypoints(waypoints, start_time, end_time):
    dense = generate_dense_waypoints(waypoints)
    timed = []

    distances = [0.0]
    total_dist = 0.0
    for i in range(1, len(dense)):
        d = euclidean(dense[i-1], dense[i])
        total_dist += d
        distances.append(total_dist)

    for i, point in enumerate(dense):
        ratio = distances[i] / total_dist if total_dist > 0 else 0
        timestamp = start_time + ratio * (end_time - start_time)
        timed.append({"point": point, "timestamp": timestamp})
    return timed

def check_spatial_conflicts(primary, others, threshold=1.0):
    conflicts = []
    for p in primary:
        for flight_id, waypoints in others.items():
            for o in waypoints:
                if euclidean(p["point"], o["point"]) <= threshold:
                    conflicts.append((p, o, flight_id))
    return conflicts

def check_temporal_conflicts(spatial_conflicts, time_threshold=5.0):
    temporal_conflicts = []
    seen = set()

    for p, o, fid in spatial_conflicts:
        t1, t2 = p["timestamp"], o["timestamp"]
        if abs(t1 - t2) <= time_threshold:
            key = (p["point"], o["point"], fid)
            if key not in seen:
                temporal_conflicts.append({
                    "conflicting_flight_id": fid,
                    "primary_point": p["point"],
                    "primary_timestamp": t1,
                    "other_point": o["point"],
                    "other_timestamp": t2
                })
                seen.add(key)
    return temporal_conflicts

def save_conflicts(conflicts, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "conflicting_flight_id",
            "primary_x", "primary_y", "primary_z", "primary_timestamp",
            "other_x", "other_y", "other_z", "other_timestamp"
        ])
        for c in conflicts:
            writer.writerow([
                c["conflicting_flight_id"],
                *map(lambda x: f"{x:.2f}", c["primary_point"]),
                f"{c['primary_timestamp']:.2f}",
                *map(lambda x: f"{x:.2f}", c["other_point"]),
                f"{c['other_timestamp']:.2f}"
            ])

def plot_conflicts_with_time(primary_timed, others_timed, spatial_conflicts, temporal_conflicts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot other flight paths first (so primary stays on top)
    for flight_id, wps in others_timed.items():
        ox, oy, oz = zip(*[p["point"] for p in wps])
        color = (random.random(), random.random(), random.random())
        ax.plot(ox, oy, oz, label=f"Flight {flight_id}", color=color)
        ax.scatter(ox, oy, oz, color=color, s=3)

    # Plot primary path in bold red
    px, py, pz = zip(*[p["point"] for p in primary_timed])
    ax.plot(px, py, pz, label="Primary Path", color='red', linewidth=3)
    ax.scatter(px, py, pz, color='red', s=3)

    # Plot temporal conflicts
    for c in temporal_conflicts:
        ax.scatter(*c["primary_point"], color='blue', marker='x', s=50, 
                   label='Temporal Conflict' if 'Temporal Conflict' not in ax.get_legend_handles_labels()[1] else "")
        ax.scatter(*c["other_point"], color='blue', marker='x', s=50)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Flight Conflict Visualization (X, Y, Z only)")
    ax.legend()
    plt.tight_layout()
    plt.show()




def main():
    primary_flight = load_flights("primary.csv")[0]
    other_flights = load_flights("others.csv")

    primary_timed = generate_timed_dense_waypoints(primary_flight["waypoints"], primary_flight["start_time"], primary_flight["end_time"])
    others_timed = {
        f["id"]: generate_timed_dense_waypoints(f["waypoints"], f["start_time"], f["end_time"]) for f in other_flights
    }

    spatial_conflicts = check_spatial_conflicts(primary_timed, others_timed)
    temporal_conflicts = check_temporal_conflicts(spatial_conflicts)

    save_conflicts(temporal_conflicts, "output.csv")
    print(f"✅ Deconfliction complete. {len(temporal_conflicts)} temporal conflicts written to output.csv")

    # Show visualization
    plot_conflicts_with_time(primary_timed, others_timed, spatial_conflicts, temporal_conflicts)

if __name__ == "__main__":
    main()
