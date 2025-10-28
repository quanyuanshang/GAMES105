import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
    stack=[]
    
    for line in lines:
        tokens=line.strip().split()
#         line.strip(): 去除字符串两端的空白字符（空格、制表符、换行符等）
# .split(): 将字符串按空白字符分割成一个列表
        if len(tokens)==0:
            continue
        if tokens[0] == "MOTION":
            break  # 层级部分结束

        elif tokens[0] == "ROOT":
            name = tokens[1]
            joint_name.append(name)
            joint_parent.append(-1)  # 根节点没有父节点
            stack.append(len(joint_name) - 1)  # 将根节点索引压入栈
        elif tokens[0] == "JOINT":
            name = tokens[1]
            joint_name.append(name)
            # 父节点是栈顶元素（当前层级的父节点）
            if len(stack) > 0:
                joint_parent.append(stack[-1])
            else:
                joint_parent.append(-1)
            stack.append(len(joint_name) - 1)  # 将当前关节索引压入栈
        elif tokens[0] == "End":
            # End Site 特殊处理
            if len(stack) > 0:
                parent_idx = stack[-1]
                end_name = joint_name[parent_idx] + "_end"
                joint_name.append(end_name)
                joint_parent.append(parent_idx)
                stack.append(len(joint_name) - 1)  # End Site也压入栈

        elif tokens[0] == "OFFSET":
            offset = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            joint_offset.append(offset)
        

        elif tokens[0] == "}":
            # 退出当前层级
            
            if len(stack) > 0:
                stack.pop()  # 弹出当前关节索引


    return joint_name, joint_parent, np.array(joint_offset)


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    M=len(joint_name)
    if joint_offset.shape[0]!=M:
        raise  ValueError("joint_offset shape mismatch")
    N=motion_data.shape[0]
    if frame_id<0 or frame_id>=N:
        raise  ValueError("frame_id out of range")
    frame=motion_data[frame_id]
    channels_per_joint=[]
    for i,name in enumerate(joint_name):
        if name.endswith("_end"):
            channels_per_joint.append(0)
        else:
            if i==0:
                channels_per_joint.append(6)
            else:
                channels_per_joint.append(3)
    total_channels=sum(channels_per_joint)
    if frame.shape[0]!=total_channels:
        raise  ValueError("motion_data shape mismatch")

    local_translations = [np.zeros(3) for _ in range(M)]
    local_eulers = [np.zeros(3) for _ in range(M)]

    idx=0
    for i,ch in enumerate(channels_per_joint):
        if ch==6:
            tx, ty, tz = frame[idx:idx+3]
            rx, ry, rz = frame[idx+3:idx+6]
            local_translations[i]=np.array([tx,ty,tz])
            local_eulers[i]=np.array([rx,ry,rz])
            idx+=6
        elif ch==3:
            rx, ry, rz = frame[idx:idx+3]
            local_eulers[i]=np.array([rx,ry,rz])
            idx+=3
        else:
            # end site
            pass
    joint_positions=np.zeros((M,3))
    joint_orientations=np.zeros((M,4))
    global_rots=[None]*M
    for i in range(M):
        parent=joint_parent[i]
        offset=np.array(joint_offset[i])
        if channels_per_joint[i]==0:
             # end site: no local rotation; orientation = parent's orientation; pos = parent_pos + parent_orient.apply(offset)
            if parent == -1:
                # weird: end site as root (shouldn't happen) -> treat as offset from origin
                parent_pos = np.zeros(3, dtype=float)
                parent_rot = R.identity()
            else:
                parent_pos = joint_positions[parent]
                parent_rot = global_rots[parent]
            pos = parent_pos + parent_rot.apply(offset)
            rot = parent_rot
        else:
            if parent == -1:
                # root joint
                root_translation = local_translations[i]
                local_rot= R.from_euler('XYZ', local_eulers[i], degrees=True)
                rot= local_rot
                pos= root_translation + offset
            else:
                parent_pos = joint_positions[parent]
                parent_rot = global_rots[parent]
                local_rot= R.from_euler('XYZ', local_eulers[i], degrees=True)
                rot= parent_rot * local_rot#BVH的旋转顺序是先局部旋转再父节点旋转
                pos= parent_pos + parent_rot.apply(offset)#将offset从父节点的局部坐标系转换到世界坐标系
        joint_positions[i, :] = pos
        joint_orientations[i, :] = rot.as_quat()  # (x, y, z, w)
        global_rots[i]=rot
    

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """

    joint_name_T, joint_parent_T, joint_offset_T = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_A, joint_parent_A, joint_offset_A = part1_calculate_T_pose(A_pose_bvh_path)
    motion_data_A = load_motion_data(A_pose_bvh_path)
    root_position = motion_data_A[:, :3]
    motion_data_A_rot = motion_data_A[:, 3:]  # 剩下都是关节旋转
    joint_remove_A = [j for j in joint_name_A if "_end" not in j]
    joint_remove_T = [j for j in joint_name_T if "_end" not in j]#因为 End Site 不需要旋转，所以过滤掉。

    # 5. 拆分 motion_data 成字典
    motion_dict = {}
    for idx, name in enumerate(joint_remove_A):
        motion_dict[name] = motion_data_A_rot[:, 3*idx:3*(idx+1)].copy()
    for idx,name in enumerate(joint_remove_T):
        if name =="lShoulder":
            motion_dict[name][:,2]=motion_dict[name][:,2]-45
        elif name =="rShoulder":
            motion_dict[name][:,2]=motion_dict[name][:,2]+45


    motion_data_rot = np.zeros_like(motion_data_A_rot)
    for idx, name in enumerate(joint_remove_T):
        if name in motion_dict:
            motion_data_rot[:, 3*idx:3*(idx+1)] = motion_dict[name]
    motion_data = np.concatenate([root_position, motion_data_rot], axis=1)
    return motion_data
    



    
   

    return motion_data
