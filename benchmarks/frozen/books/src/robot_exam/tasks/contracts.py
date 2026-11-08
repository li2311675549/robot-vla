"""State contracts shared by the runtime and the human-readable task report."""
CONTRACTS = {
    'INITIALIZE': ('model, seed, saved home', 'settle and validate model', 'finite stable state', 20, 'FAILURE'),
    'NAVIGATE': ('map, current odometry, goal', 'A* and wheel tracking', 'position and yaw tolerances', 240, 'replan at most twice'),
    'OBSERVE': ('RGB, depth, calibrated robot camera', 'color/PCA metric estimates', 'required detections present', 15, 'reobserve or FAILURE'),
    'SELECT_TARGET': ('detections, remaining targets', 'choose next target', 'one valid target', 10, 'FAILURE'),
    'PLAN': ('target, robot geometry and joint state', 'width check and multi-pose IK', 'at least one feasible candidate', 30, 'alternate pose or FAILURE'),
    'APPROACH': ('planned pose and joint state', 'bounded arm trajectory', 'tracking tolerance', 90, 'bounded retry or FAILURE'),
    'GRASP': ('target contact witnesses', 'close force-limited gripper', 'bilateral contacts', 10, 'at most three fruit attempts'),
    'MANIPULATE': ('grasp or passive joint constraint', 'lift, arc or slide', 'height or joint progress', 180, 'stop on tracking/contact failure'),
    'PLACE': ('bin or drawer pose', 'lower, release and retreat', 'stable released object', 90, 'FAILURE'),
    'VERIFY': ('independent scorer and settled state', 'strict physical containment', 'all requested predicates', 20, 'FAILURE'),
    'RECOVER': ('failure category and retry budget', 'reobserve, back off or replan', 'budget remains and new plan', 90, 'FAILURE when exhausted'),
    'RETURN_HOME': ('saved initial pose and map', 'A* wheel return', 'home position and yaw tolerances', 240, 'replan at most twice'),
    'SUCCESS': ('complete independent task score', 'persist results', 'all task conditions true', 5, 'terminal'),
    'FAILURE': ('exception, robot and target state', 'persist traceback and screenshot', 'evidence written', 5, 'terminal'),
}


def describe():
    return {name: dict(zip(('inputs', 'operation', 'success', 'timeout_seconds', 'recovery'), values),
                       failure='timeout, nonfinite simulation, failed physical predicate or exception',
                       next_states=['SUCCESS', 'FAILURE'] if name == 'VERIFY' else
                       ([] if name in ('SUCCESS', 'FAILURE') else ['task-defined next state', 'RECOVER', 'FAILURE']))
            for name, values in CONTRACTS.items()}
