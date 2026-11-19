import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from rusim.controllers import Controller
from rusim.mujoco_backend import make_model
from rusim.patient import PatientMotion
from rusim.safety import SafetyMonitor
from rusim.tissue import TissueModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default="results/animation")
    parser.add_argument("--frames",type=int,default=144)
    parser.add_argument("--fps",type=int,default=12)
    parser.add_argument("--controller",choices=("impedance","pi","sta","blsmc","blf_stsmc"),default="blf_stsmc")
    args=parser.parse_args()
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    model=make_model()
    data=mujoco.MjData(model)
    qref=np.array([0.09803821,0.53355174,-0.39410614,1.2465079,
                   0.26504926,0.72949134,-0.08275502])
    data.qpos[:]=qref
    data.ctrl[:]=qref
    mujoco.mj_forward(model,data)
    site_id=model.site("attachment_site").id
    tissue_geom_id=model.geom("tissue_surface_reference").id
    z0=float(data.site_xpos[site_id,2])
    motion=PatientMotion("liver",0)
    rng=np.random.default_rng(7)
    controller=Controller(args.controller,gains={"kp":.009,"ki":.25,"k1":1.2,"k2":3.0,
                                                  "barrier_margin":4.0,"barrier_gain":2.0,
                                                  "switch_gain":1.5,"blf_st_k1":1.5,
                                                  "blf_st_k2":2.5})
    monitor=SafetyMonitor()
    tissue=TissueModel()
    renderer=mujoco.Renderer(model,height=450,width=600)
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=np.array([0.35,0.0,0.55])
    camera.distance=1.25
    camera.azimuth=135.0
    camera.elevation=-12.0
    duration=12.0
    dt=float(model.opt.timestep)
    frame_times=np.linspace(0.0,duration,args.frames)
    steps_per_frame=max(1,int(round((duration/max(1,args.frames-1))/dt)))
    qvel_prev=data.qvel.copy()
    force_rows=[]
    frames=[]
    command_tau=np.zeros(model.nv)
    for frame_time in frame_times:
        for _ in range(steps_per_frame):
            t=float(data.time)
            base=z0-.010+.008*min(t/1.0,1.0)
            surface=base+motion.displacement(t)
            surface_velocity=motion.velocity(t)
            mujoco.mj_forward(model,data)
            position=float(data.site_xpos[site_id,2])
            model.geom_pos[tissue_geom_id,:]=np.array([
                data.site_xpos[site_id,0],
                data.site_xpos[site_id,1],
                surface-0.0021,
            ])
            jacobian=np.zeros((3,model.nv))
            mujoco.mj_jacSite(model,data,jacobian,None,site_id)
            penetration=max(0.0,position-surface)
            true_force=tissue.normal_force(penetration,surface_velocity)
            qacc=(data.qvel-qvel_prev)/dt
            qvel_prev=data.qvel.copy()
            tissue_tau=-jacobian[2]*true_force
            mass=np.zeros((model.nv,model.nv))
            mujoco.mj_fullM(model,mass,data.qM)
            tau_sensor=mass@qacc+data.qfrc_bias-tissue_tau
            dynamic_rhs=mass@qacc+data.qfrc_bias
            estimate=float(-np.dot(jacobian[2],dynamic_rhs-tau_sensor)/(np.dot(jacobian[2],jacobian[2])+1e-9))
            state=monitor.update(estimate,true_force+rng.normal(0,.12),dt)
            if int(round(t/dt)) % 2 == 0:
                normal=controller.command(state["estimate"],2*dt,surface_velocity,False)
                posture=data.qfrc_bias-8.*(data.qpos-qref)-2.*data.qvel
                command_tau=jacobian[2]*normal+posture
                command_tau=np.clip(command_tau,
                                    -np.array([176,176,110,110,110,40,40]),
                                    np.array([176,176,110,110,110,40,40]))
                data.ctrl[:]=qref
            data.qfrc_applied[:]=tissue_tau+command_tau
            mujoco.mj_step(model,data)
        renderer.update_scene(data,camera=camera)
        frame=np.asarray(renderer.render())
        image=Image.fromarray(frame)
        draw=ImageDraw.Draw(image)
        latest=force_rows[-1] if force_rows else (0.0,0.0,0.0,0.0)
        draw.rectangle((12,12,470,94),fill=(255,255,255,215))
        draw.text((24,22),f"MuJoCo force-control rendering: {args.controller.upper()}",fill=(20,45,65))
        draw.text((24,48),f"t={frame_time:.2f} s  true={latest[1]:.2f} N  estimate={latest[2]:.2f} N",fill=(20,45,65))
        draw.text((24,70),f"commanded normal force={latest[3]:.2f} N",fill=(20,45,65))
        frames.append(np.asarray(image))
        force_rows.append((frame_time,true_force,estimate,normal))
    frames=np.asarray(frames)
    telemetry=np.asarray(force_rows)
    fig,(image_axis,plot_axis)=plt.subplots(1,2,figsize=(10,4.5),gridspec_kw={"width_ratios":[1.45,1]})
    image_artist=image_axis.imshow(frames[0])
    image_axis.axis("off")
    image_axis.set_title("MuJoCo-rendered iiwa")
    true_line,=plot_axis.plot([],[],color="#d95f02",label="true force")
    estimate_line,=plot_axis.plot([],[],color="#2878b5",label="estimated force")
    command_line,=plot_axis.plot([],[],color="#31a354",label="commanded force")
    target_line,=plot_axis.plot(frame_times,np.full(len(frame_times),10.),"--",color="black",label="target")
    plot_axis.set_xlim(0,duration)
    plot_axis.set_ylim(0,18)
    plot_axis.set_xlabel("Time (s)")
    plot_axis.set_ylabel("Force (N)")
    plot_axis.set_title("Controller signals")
    plot_axis.grid(alpha=.25)
    plot_axis.legend(fontsize=8)
    def update(index):
        image_artist.set_data(frames[index])
        history=telemetry[:index+1]
        true_line.set_data(history[:,0],history[:,1])
        estimate_line.set_data(history[:,0],history[:,2])
        command_line.set_data(history[:,0],history[:,3])
        return image_artist,true_line,estimate_line,command_line,target_line
    animation=FuncAnimation(fig,update,frames=len(frames),interval=1000/args.fps,blit=True)
    animation.save(output_dir/"robot_breathing_contact.gif",writer=PillowWriter(fps=args.fps))
    animation.save(output_dir/"robot_breathing_contact.mp4",writer=FFMpegWriter(fps=args.fps,bitrate=2200))
    np.savetxt(output_dir/"robot_breathing_force_control.csv",telemetry,
               delimiter=",",header="time_s,true_force_N,estimated_force_N,commanded_normal_force_N",comments="")
    plt.close(fig)


if __name__=="__main__":
    main()
