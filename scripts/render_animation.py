import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
import numpy as np
import mujoco

from rusim.patient import PatientMotion
from rusim.mujoco_backend import make_model


def body_positions(model, data):
    return np.array(data.xpos[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/animation")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    data = mujoco.MjData(model)
    qref = np.array([0.0, 0.5, 0.0, 1.0, 0.0, -1.0, 0.0])
    data.qpos[:] = qref
    mujoco.mj_forward(model, data)
    site_id = model.site("attachment_site").id
    z0 = float(data.site_xpos[site_id, 2])
    motion = PatientMotion("liver", 0)
    times = np.linspace(0.0, 8.0, args.frames)
    poses = []
    tool_points = []
    surfaces = []
    for t in times:
        q = qref + 0.08*np.sin(2*np.pi*t/8.0 + np.arange(7)*0.45)
        q[1] += 0.04*np.sin(2*np.pi*t/motion.breathing_period)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        poses.append(body_positions(model, data))
        tool_points.append(data.site_xpos[site_id].copy())
        surfaces.append(z0 - 0.004 + motion.breathing_displacement(t))
    poses = np.asarray(poses)
    tool_points = np.asarray(tool_points)
    surfaces = np.asarray(surfaces)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.9, 0.9)
    ax.set_zlim(0.0, 1.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("KUKA iiwa robotic ultrasound contact under breathing motion")
    lines = [ax.plot([], [], [], color="#174a6e", linewidth=3)[0]]
    tool, = ax.plot([], [], [], "o", color="#d95f02", markersize=7)
    tissue, = ax.plot([], [], [], color="#2ca25f", linewidth=3, label="moving tissue surface")
    trace, = ax.plot([], [], [], color="#d95f02", linewidth=1.5, alpha=0.8, label="probe trace")
    force_arrow, = ax.plot([], [], [], color="#8c2d04", linewidth=2, label="contact direction")
    time_text = ax.text2D(0.03, 0.94, "", transform=ax.transAxes)
    ax.legend(loc="upper left")

    def update(index):
        data_points = poses[index]
        parent_points = np.vstack([np.zeros((1, 3)), data_points[:-1]])
        line_points = np.column_stack([parent_points, data_points]).reshape(-1, 3)
        line_x = line_points[:, 0]
        line_y = line_points[:, 1]
        line_z = line_points[:, 2]
        lines[0].set_data(line_x, line_y)
        lines[0].set_3d_properties(line_z)
        p = tool_points[index]
        tool.set_data([p[0]], [p[1]])
        tool.set_3d_properties([p[2]])
        tissue.set_data([-.65, .65], [0.0, 0.0])
        tissue.set_3d_properties([surfaces[index], surfaces[index]])
        history = tool_points[:index+1]
        trace.set_data(history[:, 0], history[:, 1])
        trace.set_3d_properties(history[:, 2])
        force_arrow.set_data([p[0], p[0]], [p[1], p[1]])
        force_arrow.set_3d_properties([p[2], surfaces[index]])
        time_text.set_text(f"t = {times[index]:.2f} s   breathing period = {motion.breathing_period:.2f} s")
        return lines[0], tool, tissue, trace, force_arrow, time_text

    animation = FuncAnimation(fig, update, frames=args.frames, interval=1000/args.fps, blit=False)
    animation.save(output_dir / "robot_breathing_contact.gif", writer=PillowWriter(fps=args.fps))
    animation.save(output_dir / "robot_breathing_contact.mp4", writer=FFMpegWriter(fps=args.fps, bitrate=1800))
    plt.close(fig)


if __name__ == "__main__":
    main()
