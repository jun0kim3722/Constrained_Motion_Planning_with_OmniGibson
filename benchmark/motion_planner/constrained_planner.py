import torch as th
import numpy as np

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

import omnigibson as ogb
# from scipy.spatial.transform import Rotation as R
import omnigibson.utils.transform_utils as T

# ********************************** constrained problem def **********************************
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

    def __init__(self, context, trans_const, quat_const, trans_mask, rot_mask, num_const):
        super(ArmConstraint, self).__init__(6, num_const)
        self.num_const = num_const

        self.fk_solver = context.fk_solver
        self.eef_name = context.robot.eef_link_names[context.robot.default_arm]

        self.trans_const_ = trans_const
        self.quat_const_ = quat_const
        self.trans_mask_ = trans_mask
        self.rot_mask_ = rot_mask


        # if self.trans_const_ is not None:
        #     self.trans_mask_ = th.isfinite(self.trans_const_)
        # else:
        #     self.trans_mask_ = th.full((3,), False)

        # if self.quat_const_ is not None:
        #     self.rot_mask_ = th.isfinite(self.quat_const_)
        # else:
        #     self.rot_mask_ = th.full((3,), False)

    def function(self, x, out):
        trans, quat = self.fk_solver.get_link_poses_quat(x, [self.eef_name])[self.eef_name]

        if self.trans_const_ is not None:
            trans_diff = (self.trans_const_ - trans)[self.trans_mask_]
        else:
            trans_diff = th.empty(0)
        
        if self.quat_const_ is not None:
            quat_diff = T.quat_distance(self.quat_const_, quat)
            axis_diff = T.quat2axisangle(quat_diff)
            rot_diff = axis_diff[self.rot_mask_]

        else:
            rot_diff = th.empty(0)

        out[0:self.num_const] = th.cat((trans_diff, rot_diff))

    def jacobian(self, x, out):
        trans, quat = self.fk_solver.get_link_poses_quat(x, [self.eef_name])[self.eef_name]

        pos_mask = th.cat((self.trans_mask_, self.rot_mask_))
        for j in range(6):
            new_joints = x.copy()
            new_joints[j] += 1e-6
            tran_p, quat_p = self.fk_solver.get_link_poses_quat(new_joints, [self.eef_name])[self.eef_name]
            quat_diff = T.quat_distance(quat_p, quat)
            axis_diff = T.quat2axisangle(quat_diff)
            out[:, j] = (th.cat((tran_p - trans, axis_diff)) / 1e-6)[pos_mask]
        
class ArmProjection(ob.ProjectionEvaluator):

    def __init__(self, space, context, trans_const, rot_const):
        super(ArmProjection, self).__init__(space)
        self.space_ = space
        self.fk_solver = context.fk_solver
        self.eef_name = context.robot.eef_link_names[context.robot.default_arm]
        # self.trans_const_ = trans_const
        # self.rot_const_ = rot_const

        # if self.trans_const_ is not None:
        #     self.pos_mask = th.isfinite(self.trans_const_)

        # if self.rot_const_ is not None:
        #     self.rot_mask = th.isfinite(self.rot_const_)

    def getDimension(self):
        return 6

    def project(self, state, projection):
        trans, rot = self.fk_solver.get_link_poses_axisangle([state[0], state[1], state[2], state[3], state[4], state[5]], [self.eef_name])[self.eef_name]

        projection[0:3] = trans
        projection[3:6] = rot

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
    def __init__(self, context, trans_const, rot_const, trans_mask, rot_mask, num_const, tolerance,
                 spaceType="PJ", tries=50, delta=0.05, lambda_=2.0, range_=0, exploration=0.75,
                 epsilon=0.05, rho=0.25, alpha=0.39, charts=200, bias=False, no_separate=False):
        
        robot = context.robot
        self.joint_control_idx = robot.arm_control_idx[robot.default_arm]
        self.dim = len(self.joint_control_idx)

        joint_limits = zip(robot.joint_lower_limits[:self.dim].tolist(), robot.joint_upper_limits[:self.dim].tolist())
        
        self.space_ = ob.RealVectorStateSpace(0)
        for lower_bound, upper_bound in joint_limits:
            print(lower_bound, upper_bound)
            self.space_.addDimension(lower_bound, upper_bound)

        # Create constraint
        self.constraint = ArmConstraint(context, trans_const, rot_const, trans_mask, rot_mask, num_const)
        self.cp_ = ConstrainedProblem(spaceType, self.space_, self.constraint, tolerance, tries, delta, lambda_, range_,
                                      exploration, epsilon, rho, alpha, charts, bias, no_separate)
        self.cp_.css.registerProjection("ur5e", ArmProjection(self.cp_.css, context, trans_const, rot_const))

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
        if not validityChecker.isValid(start, True) or not validityChecker.isValid(goal, True):
            ogb.log.warning("Invalid Start or Goal from ArmCcontrainedPlanner")

            if not validityChecker.isValid(start, True):
                print("Start")
                for _ in range(500):
                    ogb.sim.step()

            if not validityChecker.isValid(goal, True):
                print("Goal")
                for _ in range(500):
                    ogb.sim.step()

            return None

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