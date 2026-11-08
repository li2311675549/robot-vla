"""Auxiliary scene-preservation QA, separate from original task scoring."""
import numpy as np


def fruit_scene_preservation(sim,metadata,target):
    model,data=sim.model,sim.data;g=model.geom('fruit_table_top').id
    center=data.geom_xpos[g];half=model.geom_size[g];height=float(center[2]+half[2])
    rows=[]
    for obj in metadata['objects']:
        if obj['body']==target:continue
        body=model.body(obj['body']).id;position=data.xpos[body]
        dof=int(model.jnt_dofadr[model.body_jntadr[body]])
        speed=float(np.linalg.norm(data.qvel[dof:dof+3]))
        supported=bool(np.all(abs(position[:2]-center[:2])<=half[:2]+.005)
                       and height-.005<=position[2]<=height+.12)
        rows.append({'body':obj['body'],'position':position.tolist(),'on_original_table':supported,
                     'linear_speed_m_s':speed,'quiet':speed<.02})
    return {'success':all(x['on_original_table'] and x['quiet'] for x in rows),
            'scope':'Additional demonstration QA, not a changed full-task score',
            'table_center':center.tolist(),'table_halfsize':half.tolist(),
            'margin_m':.005,'maximum_speed_m_s':.02,'non_targets':rows}
