import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from rusim.mujoco_backend import make_model
from rusim.patient import PatientMotion


def target_pose(t, qref, period):
    amplitudes=np.array([.22,.30,.24,.34,.20,.28,.24])
    phases=np.arange(7)*.55
    q=qref+amplitudes*np.sin(2*np.pi*t/8.0+phases)
    q[1]+=.10*np.sin(2*np.pi*t/period)
    q[3]+=.08*np.sin(2*np.pi*t/period+.7)
    return q


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default="results/animation")
    parser.add_argument("--frames",type=int,default=144)
    parser.add_argument("--fps",type=int,default=12)
    args=parser.parse_args()
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    model=make_model()
    data=mujoco.MjData(model)
    qref=np.array([0.0,0.5,0.0,1.0,0.0,-1.0,0.0])
    data.qpos[:]=qref
    data.ctrl[:]=qref
    mujoco.mj_forward(model,data)
    site_id=model.site("attachment_site").id
    z0=float(data.site_xpos[site_id,2])
    motion=PatientMotion("liver",0)
    renderer=mujoco.Renderer(model,height=450,width=600)
    times=np.linspace(0.0,12.0,args.frames)
    frames=[]
    tool_points=[]
    surfaces=[]
    steps_per_frame=max(1,int(round((times[-1]/max(1,args.frames-1))/model.opt.timestep)))
    for t in times:
        target=target_pose(t,qref,motion.breathing_period)
        data.ctrl[:]=target
        for _ in range(steps_per_frame):
            mujoco.mj_step(model,data)
        renderer.update_scene(data)
        frame=renderer.render()
        image=Image.fromarray(frame)
        draw=ImageDraw.Draw(image)
        breathing=float(motion.breathing_displacement(t))
        draw.rectangle((12,12,440,92),fill=(255,255,255,210))
        draw.text((24,22),"MuJoCo iiwa 7 R800 dynamic rendering",fill=(20,45,65))
        draw.text((24,48),f"t={t:.2f} s   breathing displacement={breathing*1000:.2f} mm",fill=(20,45,65))
        frames.append(np.asarray(image))
        tool_points.append(data.site_xpos[site_id].copy())
        surfaces.append(z0-0.004+breathing)
    frames=np.asarray(frames)
    fig,ax=plt.subplots(figsize=(6,4.5),dpi=100)
    image_artist=ax.imshow(frames[0])
    ax.axis("off")
    ax.set_title("MuJoCo-rendered iiwa motion with respiratory surface input")
    def update(index):
        image_artist.set_data(frames[index])
        return (image_artist,)
    animation=FuncAnimation(fig,update,frames=len(frames),interval=1000/args.fps,blit=True)
    animation.save(output_dir/"robot_breathing_contact.gif",writer=PillowWriter(fps=args.fps))
    animation.save(output_dir/"robot_breathing_contact.mp4",writer=FFMpegWriter(fps=args.fps,bitrate=2200))
    plt.close(fig)
    np.savez(output_dir/"robot_breathing_contact_trace.npz",
             time=times,tool_position=np.asarray(tool_points),surface_z=np.asarray(surfaces))


if __name__=="__main__":
    main()
