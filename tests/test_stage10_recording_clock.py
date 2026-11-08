import mujoco
from robot_exam.simulation import ROOT,Simulation
from robot_exam.evaluation.recording import VideoRecorder


def test_recorder_anchors_to_nonzero_physics_time(tmp_path):
    sim=Simulation(ROOT/'scenes/ur5_gripper.xml');sim.step(sim.data.ctrl,500)
    view=mujoco.MjvCamera();view.lookat[:]=[.4,0,.7];view.distance=2.;view.azimuth=150;view.elevation=-25
    recorder=VideoRecorder(tmp_path,camera=view,fps=5)
    try:
        recorder(sim,'checkpoint')
        for _ in range(105):sim.step(sim.data.ctrl);recorder(sim,'checkpoint')
        result=recorder.close()
        assert result['status']=='PASS'
        assert result['cadence_validated']
        assert result['frames']==2
        assert result['first_sim_seconds']>=1.
        assert result['last_sim_seconds']-result['first_sim_seconds']>=.199
    finally:
        if recorder.encoder and recorder.encoder.poll() is None:recorder.close()
