import torch as th
import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

import omnigibson as ogb
# from scipy.spatial.transform import Rotation as R
import omnigibson.utils.transform_utils as T

# ********************************** constrained problem def **********************************
def getAtlasOptions():
    print("Pulling AtlasOPtions")
    return (
        ob.ATLAS_STATE_SPACE_EPSILON,
        ob.CONSTRAINED_STATE_SPACE_DELTA * ob.ATLAS_STATE_SPACE_RHO_MULTIPLIER,
        ob.ATLAS_STATE_SPACE_EXPLORATION,
        ob.ATLAS_STATE_SPACE_ALPHA,
        ob.ATLAS_STATE_SPACE_MAX_CHARTS_PER_EXTENSION,
        False,
        True,
    )

class ConstrainedProblem(object):

    def __init__(self, spaceType, space, constraint, tolerance, tries, delta, lambda_, range_,
                 exploration, epsilon, rho, alpha, charts, bias, no_separate):
        self.spaceType = spaceType
        self.space = space
        self.constraint = constraint
        self.constraint.setTolerance(tolerance)
        self.constraint.setMaxIterations(tries)
        self.range = range_
        self.pp = None

        if spaceType == "PJ":
            ou.OMPL_INFORM("Using Projection-Based State Space!")
            self.css = ob.ProjectedStateSpace(space, constraint)
            self.csi = ob.ConstrainedSpaceInformation(self.css)
        elif spaceType == "AT":
            ou.OMPL_INFORM("Using Atlas-Based State Space!")
            self.css = ob.AtlasStateSpace(space, constraint)
            self.csi = ob.ConstrainedSpaceInformation(self.css)
        elif spaceType == "TB":
            ou.OMPL_INFORM("Using Tangent Bundle-Based State Space!")
            self.css = ob.TangentBundleStateSpace(space, constraint)
            self.csi = ob.TangentBundleSpaceInformation(self.css)
        else:
            assert False, "spaceType should be one of the following. 'PJ', 'AT', 'TB'"

        self.css.setup()
        self.css.setDelta(delta)
        self.css.setLambda(lambda_)
        if not spaceType == "PJ":
            self.css.setExploration(exploration)
            self.css.setEpsilon(epsilon)
            self.css.setRho(rho)
            self.css.setAlpha(alpha)
            self.css.setMaxChartsPerExtension(charts)
            if bias:
                self.css.setBiasFunction(lambda c, atlas=self.css:
                                         atlas.getChartCount() - c.getNeighborCount() + 1.)
            if spaceType == "AT":
                self.css.setSeparated(not no_separate)
            self.css.setup()
        self.ss = og.SimpleSetup(self.csi)

    def setStartAndGoalStates(self, start, goal):
        # Create start and goal states
        if self.spaceType == "AT" or self.spaceType == "TB":
            self.css.anchorChart(start())
            self.css.anchorChart(goal())

        # Setup problem
        self.ss.setStartAndGoalStates(start, goal)

    def getPlanner(self, plannerName, projectionName=None):
        planner = eval('og.%s(self.csi)' % plannerName)
        try:
            if self.options.range == 0:
                if not self.spaceType == "PJ":
                    planner.setRange(self.css.getRho_s())
            else:
                planner.setRange(self.options.range)
        except:
            pass
        try:
            if projectionName:
                planner.setProjectionEvaluator(projectionName)
        except:
            pass
        return planner

    def setPlanner(self, plannerName, projectionName=None):
        self.pp = self.getPlanner(plannerName, projectionName)
        self.ss.setPlanner(self.pp)


# ********************************** constrained planner **********************************
class ArmConstraint(ob.Constraint):

    def __init__(self, robot_dof, num_const, custom_fn):
        super(ArmConstraint, self).__init__(robot_dof, num_const)
        self.num_const = num_const
        self.constraint_fn = custom_fn

    def function(self, x, out):
        out[0:self.num_const] = self.constraint_fn(x)

    def jacobian(self, x, out):
        n = x.shape[0]
        for i in range(n):
            dx = np.zeros_like(x)
            dx[i] = 1e-6
            out[:, i] = ((self.constraint_fn(x + dx) - self.constraint_fn(x - dx)) / (2 * 1e-6)).detach().numpy()

class ArmProjection(ob.ProjectionEvaluator):

    def __init__(self, space, context):
        super(ArmProjection, self).__init__(space)
        self.space_ = space
        self.fk_solver = context.fk_solver
        self.eef_name = context.robot.eef_link_names[context.robot.default_arm]

    def getDimension(self):
        return 3

    def project(self, state, projection):
        trans, rot = self.fk_solver.get_link_poses_axisangle([state[0], state[1], state[2], state[3], state[4], state[5]], [self.eef_name])[self.eef_name]
        projection[0:3] = trans
        # projection[3:6] = rot

class ArmValidAll(ob.StateValidityChecker):
    def __init__(self, si, context):
        super().__init__(si)
        self.context = context
        robot = context.robot
        self.dim = len(robot.arm_control_idx[robot.default_arm])

    def isValid(self, dof_state, debug=False):
        joint_pos = th.tensor([dof_state[i] for i in range(self.dim)])
        return not self.context.set_arm_and_detect_collision(joint_pos, debug)

class ArmCcontrainedPlanner():
    def __init__(self, context, tolerance=0.1, custom_fn=None, num_const=None,
                 spaceType="PJ", tries=50, delta=0.05, lambda_=2.0, range_=0, exploration=0.75,
                 epsilon=0.05, rho=0.25, alpha=0.39, charts=200, bias=False, no_separate=False):
        
        robot = context.robot
        self.joint_control_idx = robot.arm_control_idx[robot.default_arm]
        self.dim = len(self.joint_control_idx)

        joint_limits = zip(robot.joint_lower_limits[:self.dim].tolist(), robot.joint_upper_limits[:self.dim].tolist())
        
        self.space_ = ob.RealVectorStateSpace(0)
        for lower_bound, upper_bound in joint_limits:
            self.space_.addDimension(lower_bound, upper_bound)
        
        if spaceType=="AT":
            (
                epsilon,
                rho,
                exploration,
                alpha,
                charts,
                bias,
                no_separate,
            ) = getAtlasOptions()

        # Create constraint
        self.constraint = ArmConstraint(self.dim, num_const, custom_fn)
        self.cp_ = ConstrainedProblem(spaceType, self.space_, self.constraint, tolerance, tries, delta, lambda_, range_,
                                      exploration, epsilon, rho, alpha, charts, bias, no_separate)
        if spaceType=="PJ":
            self.cp_.css.registerProjection("ur5e", ArmProjection(self.cp_.css, context))

    def plan(self, start_conf, end_conf, context, planner_type="KPIECE1", planning_time=30.0):
        start = ob.State(self.cp_.css)
        for i in range(self.dim):
            start[i] = float(start_conf[i])

        goal = ob.State(self.cp_.css)
        for i in range(self.dim):
            goal[i] = float(end_conf[i])

        si = self.cp_.ss.getSpaceInformation()
        validityChecker = ArmValidAll(si, context)
        si.setStateValidityChecker(validityChecker)
        si.setStateValidityCheckingResolution(0.0005)

        # check start and goal state
        # assert validityChecker.isValid(start) and validityChecker.isValid(goal), "Invalid Start or Goal"
        if not validityChecker.isValid(start, True):
            ogb.log.warning("Invalid Start from ArmCcontrainedPlanner")
        
        if not validityChecker.isValid(goal, True):
            ogb.log.warning("Invalid Goal from ArmCcontrainedPlanner")

            # if not validityChecker.isValid(start, True):
            #     print("Start")
            #     breakpoint()
            #     for _ in range(500):
            #         ogb.sim.step()

            # if not validityChecker.isValid(goal, True):
            #     breakpoint()
            #     print("Goal")
            #     for _ in range(500):
            #         ogb.sim.step()

            return None
        
        # Check Start & Goal constraint
        st_result = np.zeros(self.constraint.num_const)
        self.constraint.function(start_conf, st_result)
        if (st_result > 0.1).any():
            print("Start does not meet constraint")
            breakpoint()

        gl_result = np.zeros(self.constraint.num_const)
        self.constraint.function(end_conf, gl_result)
        if (gl_result > 0.1).any():
            print("Goal does not meet constraint")
            breakpoint()

        self.cp_.setStartAndGoalStates(start, goal)
        self.cp_.setPlanner(planner_type, "ur5e")

        # Solve the problem
        temp_res = self.cp_.ss.solve(planning_time)
        if temp_res.asString() == 'Exact solution':
            ou.OMPL_DEBUG("Exact solution found")
            path = self.cp_.ss.getSolutionPath()

            path_simp = og.PathSimplifier(si)
            res = path_simp.reduceVertices(path)

            path_list = []
            for t in range(path.getStateCount()):
                state = path.getState(t)
                path_list.append([state[0], state[1], state[2], state[3], state[4], state[5]])

            return path_list
        else:
            return None