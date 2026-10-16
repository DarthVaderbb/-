"""教学用一维控制仿真；不连接真实设备。"""
from pathlib import Path
import csv
import argparse


def simulate(mode="PD", kp=4.0, kd=2.0, ki=0.0,
             limit=None, disturbance=0.0, duration=30.0):
    mass, damping, dt, target = 1.0, 1.0, 0.01, 1.0
    x, v, integral = 0.0, 0.0, 0.0
    rows = []
    for step in range(round(duration / dt)):
        t = step * dt
        error = target - x
        external = disturbance if t >= 5.0 else 0.0
        if mode == "OPEN":
            requested = 1.0 if t < 2.0 else 0.0
        else:
            # I项先累积误差；积分限幅只是入门保护，不是完整抗饱和设计。
            if mode == "PID":
                integral = max(-5.0, min(5.0, integral + error * dt))
            requested = kp * error
            if mode in ("PD", "PID"):
                requested -= kd * v
            if mode == "PID":
                requested += ki * integral
        applied = requested
        if limit is not None:
            applied = max(-limit, min(limit, requested))
        # 本行状态与本行施加的输入对应同一时刻；更新发生在记录之后。
        rows.append((t, x, v, target, error, requested, applied, external))
        a = (applied + external - damping * v) / mass
        v = v + a * dt
        x = x + v * dt
    return rows


def cases_for(week):
    if week == 5:
        return [("Open loop: force removed at 2 s", dict(mode="OPEN"))]
    if week == 6:
        return [(f"P: Kp={k}", dict(mode="P", kp=k)) for k in (1, 4, 9)]
    if week == 7:
        return [(f"PD: Kd={k}", dict(kp=4, kd=k)) for k in (0, 2, 4)] + [
            ("PD: Kp=9 unlimited", dict(kp=9, kd=2)),
            ("PD: Kp=9 limit=5 N", dict(kp=9, kd=2, limit=5))]
    return [
        ("PD: disturbance=1 N", dict(kp=4, kd=2, disturbance=1)),
        ("PID: Ki=0.5", dict(mode="PID", kp=4, kd=2, ki=0.5, disturbance=1)),
        ("PID: Ki=1", dict(mode="PID", kp=4, kd=2, ki=1, disturbance=1)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, choices=(5, 6, 7, 8), default=5)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    out = Path(__file__).resolve().parent / f"week{args.week}_results"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for index, (label, settings) in enumerate(cases_for(args.week), 1):
        rows = simulate(**settings)
        with (out / f"case{index}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(("time_s", "position_m", "velocity_m_s", "target_m",
                             "error_m", "requested_force_N", "applied_force_N", "disturbance_N"))
            writer.writerows(rows)
        axes[0].plot([r[0] for r in rows], [r[1] for r in rows], label=label)
        axes[1].plot([r[0] for r in rows], [r[6] for r in rows], label=label)
        tail = [r for r in rows if r[0] >= 25]
        mae = sum(abs(r[4]) for r in tail) / len(tail)
        overshoot = max(0.0, max(r[1] for r in rows) - 1.0)
        peak_force = max(abs(r[6]) for r in rows)
        print(f"{label}: last-5s MAE={mae:.5f} m; overshoot={overshoot:.5f} m; peak force={peak_force:.2f} N")
    if args.week != 5:
        axes[0].axhline(1, color="black", linestyle="--", label="Target")
    axes[0].set_ylabel("Position (m)")
    axes[1].set_ylabel("Applied control force (N)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    print("Results:", out)
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
