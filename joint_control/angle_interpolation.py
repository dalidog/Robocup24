'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.start_time = None
        self.animation_done = True

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        #target_joints['RHipYawPitch'] = target_joints['LHipYawPitch'] # copy missing joint in keyframes
        #target_joints['RHipYawPitch'] = target_joints.get('LHipYawPitch', 0)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def bezier(self, t, p0, p1, p2, p3):
        p0 = p0*(1 - t)**3
        p1 = p1*3*t*(1 - t)**2
        p2 = p2*3*t ** 2*(1 - t)
        p3 = p3*t**3
        return p0+p1+p2+p3

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        # YOUR CODE HERE
        names, times, keys = keyframes
        self.start_time = perception.time if (self.start_time is None) else self.start_time
        dt = perception.time - self.start_time
        for i, tmpname in enumerate(names):
            if tmpname == 'RHipYawPitch':
                tmpname = 'LHipYawPitch'
            if tmpname not in self.joint_names:
                continue
            tmpkey = keys[i]
            tmptime = times[i]
            for count, t in enumerate(tmptime[:-1]):
                if t < dt < tmptime[count + 1]:
                    p0 = tmpkey[count][0]
                    p1 = p0 + (tmpkey[count][2][1] * tmpkey[count][2][2])
                    p2 = tmpkey[count + 1][0]
                    p3 = p2 + (tmpkey[count + 1][1][1] * tmpkey[count + 1][1][2])
                    dT = (dt - t) / (tmptime[count + 1] - t)
                    target_joints[tmpname] = self.bezier(dT, p0, p1, p2, p3)
                elif tmptime[0] > dt:
                    p0 = perception.joint[tmpname]
                    p1 = p0
                    p2 = tmpkey[0][0]
                    p3 = p2 + (tmpkey[count][2][1] * tmpkey[count][2][2])
                    dT = dt  / tmptime[1]
                    target_joints[tmpname] = self.bezier(dT, p0, p1, p2, p3)
        self.animation_done = target_joints == {}
        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    print('hello')
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()