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

def rotation_matrix(a, b):
    """
    返回将向量a旋转到向量b的旋转矩阵
    稳健版本：处理小向量、平行、反平行，保证右手坐标系
    """
    # 小向量直接返回单位矩阵
    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return np.eye(3)

    # 单位化
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    # 平行
    if np.allclose(a, b):
        return np.eye(3)

    # 反平行
    if np.allclose(a, -b):
        # 找一个与a不平行的轴
        if abs(a[0]) < 0.9:
            perp = np.array([1, 0, 0])
        else:
            perp = np.array([0, 1, 0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    # 计算旋转轴和角度
    n = np.cross(a, b)
    sin_theta = np.linalg.norm(n)
    cos_theta = np.dot(a, b)
    n = n / sin_theta  # 旋转轴单位化

    c = cos_theta
    s = sin_theta
    v = 1 - c

    R_mat = np.array([
        [n[0]*n[0]*v + c,   n[0]*n[1]*v - n[2]*s, n[0]*n[2]*v + n[1]*s],
        [n[0]*n[1]*v + n[2]*s, n[1]*n[1]*v + c,   n[1]*n[2]*v - n[0]*s],
        [n[0]*n[2]*v - n[1]*s, n[1]*n[2]*v + n[0]*s, n[2]*n[2]*v + c]
    ])

    # SVD正交化，保证右手坐标系
    U, _, Vt = np.linalg.svd(R_mat)
    R_mat = U @ Vt
    if np.linalg.det(R_mat) < 0:
        R_mat[:, -1] *= -1

    return R_mat



def inv_safe(data):
    # return R.from_quat(data).inv()
    if np.allclose(data, [0, 0, 0, 0]):
        return np.eye(3)#如果四元数是 [0,0,0,0]（无效四元数）→ 返回3x3单位矩阵
    else:
        return np.linalg.inv(R.from_quat(data).as_matrix())#将四元数转为旋转矩阵，然后求逆
    
def from_quat_safe(data):
    # return R.from_quat(data)
    if np.allclose(data, [0, 0, 0, 0]):
        return np.eye(3)
    else:
        return R.from_quat(data).as_matrix()
    


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path,path_name,path1,path2=meta_data.get_path_from_root_to_end()
    parent_idx = meta_data.joint_parent
    no_caled_orientation=joint_orientations.copy()
    local_rotation = [
    R.from_matrix(inv_safe(joint_orientations[parent_idx[i]]) @ from_quat_safe(joint_orientations[i])).as_quat()
    for i in range(len(joint_orientations))]
    local_rotation[0]=R.from_matrix(from_quat_safe(joint_orientations[0])).as_quat()#root
    local_position= [joint_positions[i]-joint_positions[parent_idx[i]] for i
                      in range(len(joint_orientations))]
    local_position[0]=joint_positions[0]#root

    path_end_id=path1[0]#hand
    for k in range(0,300):
        # k：循环次数
        # 正向的，path1是从手到root之前
        for idx in range(0,len(path1)):
            # idx：路径上的第几个节点了，第0个是手，最后一个是root
            path_joint_id=path1[idx]

            vec_to_end=joint_positions[path_end_id]-joint_positions[path_joint_id]
            vec_to_target=target_pose-joint_positions[path_joint_id]
            # 获取end->target的旋转矩阵
            rot_matrix=rotation_matrix(vec_to_end,vec_to_target)

            # 计算前的朝向。这个朝向实际上是累乘到父节点的
            initial_orientation=from_quat_safe(joint_orientations[path_joint_id])
            # 旋转矩阵，格式换算
            rot_matrix_R=R.from_matrix(rot_matrix).as_matrix()
            # 计算后的朝向
            calculated_orientation=rot_matrix_R.dot(initial_orientation)
            # 写回结果列表
            joint_orientations[path_joint_id]=R.from_matrix(calculated_orientation).as_quat()

            # 子节点的朝向也会有所变化
            # idx-1 就是当前节点的下一个更接近尾端的节点，一直向前迭代到1
            for i in range(idx-1,0,-1):
                path_joint_id_1=path1[i]
                # 这段代码是在更新子节点的朝向，确保当父节点旋转时，所有子节点也跟着旋转。
                joint_orientations[path_joint_id_1]=R.from_matrix(rot_matrix_R.dot(from_quat_safe(joint_orientations[path_joint_id_1]))).as_quat()
            # 前面的代码只更新了关节的朝向，但关节的位置还是旧的。
            for i in range(idx-1,-1,-1):
                # path_joint_id=path1[i]
                # 节点id
                next_joint_id=path1[i]
                # 指向下个节点的向量
                vec_to_next=joint_positions[next_joint_id]-joint_positions[path_joint_id]
                # 左乘，改变向量
                calculated_vec_to_next_dir=rot_matrix.dot(vec_to_next)
                # 防止长度不对
                calculated_vec_to_next=calculated_vec_to_next_dir/np.linalg.norm(calculated_vec_to_next_dir)*np.linalg.norm(vec_to_next)
                # 还原回去
                joint_positions[next_joint_id]=calculated_vec_to_next+joint_positions[path_joint_id]

            # path2是从脚到root，所以要倒着
        # debug
        # for idx in range(len(path2)-1,len(path2)-3,-1): # len(path2)-1 --> 0
        for idx in range(len(path2)-1,0,-1): # len(path2)-1 --> 0
            path_joint_id=path2[idx]
            parient_joint_id=max(parent_idx[path_joint_id],0)

            vec_to_end=joint_positions[path_end_id]-joint_positions[path_joint_id]
            vec_to_target=target_pose-joint_positions[path_joint_id]
            rot_matrix=rotation_matrix(vec_to_end,vec_to_target)

            # 计算前的朝向。注意path2是反方向的，要改父节点才行
            initial_orientation=from_quat_safe(joint_orientations[path_joint_id])
            # 旋转矩阵，格式换算
            rot_matrix_R= R.from_matrix(rot_matrix).as_matrix()
            # 计算后的朝向
            calculated_orientation=rot_matrix_R.dot(initial_orientation)
            # 写回结果列表
            joint_orientations[path_joint_id]=R.from_matrix(calculated_orientation).as_quat()

            # 其他节点的朝向也会有所变化
            for i in range(idx+1,len(path2)):
                path_joint_id=path2[i] 
                joint_orientations[path_joint_id]=R.from_matrix(rot_matrix_R.dot(from_quat_safe(joint_orientations[path_joint_id]))).as_quat()

            # idx-1 就是当前节点的下一个更接近尾端的节点，一直向前迭代到1
            for i in range(len(path1)-1,0,-1):
                path_joint_id=path1[i]
                # 遍历路径后的节点,都乘上旋转
                joint_orientations[path_joint_id]=R.from_matrix(rot_matrix_R.dot(from_quat_safe(joint_orientations[path_joint_id]))).as_quat()

            path_joint_id=path2[max(idx-1,0)]
            # 修改父节点，或者说更靠近手的那些节点的位置
            # path2上的
            for i in range(idx,len(path2)):
                # path_joint_id=path1[i]
                # 节点id
                prev_joint_id=path2[i]
                # 指向上一个节点的向量
                vec_to_next=joint_positions[prev_joint_id]-joint_positions[path_joint_id]
                # 左乘，改变向量
                calculated_vec_to_next_dir=rot_matrix.dot(vec_to_next)
                # 防止长度不对
                calculated_vec_to_next=calculated_vec_to_next_dir/np.linalg.norm(calculated_vec_to_next_dir)*np.linalg.norm(vec_to_next)
                # 还原回去
                joint_positions[prev_joint_id]=joint_positions[path_joint_id]+calculated_vec_to_next
            # path1上的
            for i in range(len(path1)-1,-1,-1):
                # path_joint_id=path1[i]
                # 节点id
                prev_joint_id=path1[i]
                # 指向上一个节点的向量
                vec_to_next=joint_positions[prev_joint_id]-joint_positions[path_joint_id]
                # 左乘，改变向量
                calculated_vec_to_next_dir=rot_matrix.dot(vec_to_next)
                # 防止长度不对
                calculated_vec_to_next=calculated_vec_to_next_dir/np.linalg.norm(calculated_vec_to_next_dir)*np.linalg.norm(vec_to_next)
                # 还原回去
                joint_positions[prev_joint_id]=calculated_vec_to_next+joint_positions[path_joint_id]

        # debug
        # rot_matrix=rotation_matrix(np.array([1,0,0]),np.array([1,0,1]))
        # joint_orientations[0]=R.from_matrix(rot_matrix).as_quat()
        # joint_orientations[1]=R.from_matrix(rot_matrix).as_quat()
        joint_orientations[path_end_id]=joint_orientations[path1[1]]#这行代码将手部的朝向设置为与手腕的朝向相同
        cur_dis=np.linalg.norm(joint_positions[path_end_id]-target_pose)
        if cur_dis<0.01:
            break
    print("距离",cur_dis,"迭代了",k,"次")
    # 更新不在链上的节点
    for k in range(len(joint_orientations)):
        if k in path:
            pass
        elif k==0:
            # 要单独处理，不然跟节点的-1就会变成从最后一个节点开始算
            pass
        else:
            # 先获取局部旋转
            # 这里如果直接存的就是矩阵就会有问题？
            local_rot_matrix=R.from_quat(local_rotation[k]).as_matrix()
            # 再获取我们已经计算了的父节点的旋转
            parent_rot_matrix=from_quat_safe(joint_orientations[parent_idx[k]])
            # 乘起来
            re=parent_rot_matrix.dot(local_rot_matrix)
            joint_orientations[k]=R.from_matrix(re).as_quat()

            # re：父节点IK后的新旋转矩阵
# initial_o：父节点IK前的原始旋转矩阵
# delta_orientation：从原始状态到新状态的旋转变化量 delta_orientation = new_rotation × old_rotation⁻¹这个公式回答了一个问题："从旧旋转状态变换到新旋转状态，需要施加什么旋转？"
            initial_o=from_quat_safe(no_caled_orientation[parent_idx[k]])
            # 父节点的旋转*delta_orientation=子节点旋转
            # 反求delta_orientation
            delta_orientation = np.dot(re, np.linalg.inv(initial_o))
            # 父节点的位置加原本基础上的旋转
            joint_positions[k]=joint_positions[parent_idx[k]]+delta_orientation.dot(local_position[k])

    return joint_positions, joint_orientations


# def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
#     path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
#     parent_idx = meta_data.joint_parent

#     # 保存原始局部旋转和骨骼长度
#     local_rotation = [
#         R.from_matrix(inv_safe(joint_orientations[parent_idx[i]]) @ from_quat_safe(joint_orientations[i])).as_quat()
#         if i != 0 else R.from_matrix(from_quat_safe(joint_orientations[0])).as_quat()
#         for i in range(len(joint_orientations))
#     ]
#     bone_lengths = [
#         np.linalg.norm(joint_positions[i] - joint_positions[parent_idx[i]]) if i != 0 else 0
#         for i in range(len(joint_positions))
#     ]

#     # 开始迭代
#     path_end_id = path1[0]  # 末端手或脚
#     for k in range(300):
#         # 1. 正向迭代 path1（手臂/腿）
#         for idx in range(len(path1)):
#             joint_id = path1[idx]
#             vec_to_end = joint_positions[path_end_id] - joint_positions[joint_id]
#             vec_to_target = target_pose - joint_positions[joint_id]

#             rot_matrix = rotation_matrix(vec_to_end, vec_to_target)
#             rot_matrix = R.from_matrix(rot_matrix).as_matrix()

#             # 更新关节朝向
#             joint_orientations[joint_id] = R.from_matrix(rot_matrix @ from_quat_safe(joint_orientations[joint_id])).as_quat()

#             # 更新子节点位置，同时保持骨骼长度不变
#             for i in range(idx-1, -1, -1):
#                 child_id = path1[i]
#                 length = bone_lengths[child_id]
#                 direction = joint_positions[child_id] - joint_positions[joint_id]
#                 direction = rot_matrix @ direction
#                 direction = direction / np.linalg.norm(direction) * length
#                 joint_positions[child_id] = joint_positions[joint_id] + direction

#         # 2. 反向迭代 path2（腿链）
#         for idx in range(len(path2)-1, 0, -1):
#             joint_id = path2[idx]
#             parent_id = parent_idx[joint_id]
#             vec_to_end = joint_positions[path_end_id] - joint_positions[joint_id]
#             vec_to_target = target_pose - joint_positions[joint_id]
#             rot_matrix = rotation_matrix(vec_to_end, vec_to_target)
#             rot_matrix = R.from_matrix(rot_matrix).as_matrix()

#             # 更新关节朝向
#             joint_orientations[joint_id] = R.from_matrix(rot_matrix @ from_quat_safe(joint_orientations[joint_id])).as_quat()

#             # 更新子节点位置，保持骨骼长度
#             for i in range(idx, len(path2)):
#                 child_id = path2[i]
#                 length = bone_lengths[child_id]
#                 direction = joint_positions[child_id] - joint_positions[joint_id]
#                 direction = rot_matrix @ direction
#                 direction = direction / np.linalg.norm(direction) * length
#                 joint_positions[child_id] = joint_positions[joint_id] + direction

#         # 3. 检查收敛
#         cur_dis = np.linalg.norm(joint_positions[path_end_id] - target_pose)
#         if cur_dis < 0.01:
#             break

#     print("距离", cur_dis, "迭代了", k, "次")

#     # 更新其他非路径节点
#     for i in range(len(joint_orientations)):
#         if i not in path and i != 0:
#             parent_rot = from_quat_safe(joint_orientations[parent_idx[i]])
#             joint_orientations[i] = R.from_matrix(parent_rot @ R.from_quat(local_rotation[i]).as_matrix()).as_quat()
#             joint_positions[i] = joint_positions[parent_idx[i]] + parent_rot @ (joint_positions[i] - joint_positions[parent_idx[i]])

#     return joint_positions, joint_orientations



def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    path, path_name, path1, path2=meta_data.get_path_from_root_to_end()
    target_pose=joint_positions[0]+np.array([relative_x,target_height-joint_positions[0][1],relative_z])
    IK_joint_positions, IK_joint_orientations=part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose)
    for i in path:
        joint_positions[i]=IK_joint_positions[i]
        
        joint_orientations[i]=IK_joint_orientations[i]

    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations