'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity, matrix, linalg
import jax.numpy as np2
from math import atan2
import numpy as np
from jax import grad, jit
from  numpy.linalg import norm

class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = {}
        # YOUR CODE HERE

        def make_work(t):
            return [t[-1, 0],  t[-1, 1],  t[-1, 2],  atan2(t[2, 1], t[2, 2]) ]

        def error_func(j_a,target):
            Te = self.forward_kinematics(j_a)
            transformValues = [x for x in self.transforms.values()]
            transformMatrix = matrix([make_work(transformValues[-1])]).T
            e = target - transformMatrix
            return norm(e)

        for j in self.chains[effector_name]:
            joint_angles[j] = self.perception.joint[j]
        for joint_c in self.chains.values():
            for joint in joint_c:
                if joint not in joint_angles:
                    joint_angles[joint] = 0

        
        func = lambda t: error_func(t, make_work(transform))
        func_grad = jit(grad(func))

        for i in range(1000):
            e = func(joint_angles)
            d = func_grad(joint_angles)
            joint_angles -= d * 1e-2
            if e < 1e-4:
                break
        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        j_angles = self.inverse_kinematics(effector_name, transform)
        n = self.chains[effector_name]

        keys = [ [
                    [self.perception.joint[name], [3, 0, 0], [3, 0, 0]],
                    [j_angles[i], [3, 0, 0], [3, 0, 0]],
                ] for i, name in enumerate(n)
                ]


        time = [[2.0, 6.0]] * len(n)
        self.keyframes = (n, time, keys)  # the result joint angles have to fill in
        print(self.keyframes)


    #theta = random.random(N)
    #def inverse_kinematics(x_e, y_e, z_e, theta_e, theta):
        #target = trans(x_e, y_e, z_e, theta_e)
        #func = lambda t: error_func(t, target)
        #func_grad = jit(grad(func))
        
        # for i in range(1000):
        #     e = func(theta)
        #     d = func_grad(theta)
        #     theta -= d * 1e-2
        #     if e < 1e-4:
        #         break
        
       # return theta

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = np.eye(4)
    #T.at[-1,1].set(0.05)
    #T.at[-1,2].set(-0.26)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
