from __future__ import annotations
import os 
import io
import numpy as np 
from itertools import product
from pydantic import dataclasses, validator
from transforms3d import affines, quaternions, euler
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt 
import open3d as o3d
from matplotlib.patches import Patch
from PIL import Image 
import seaborn as sns
Pixel = Tuple[int, int]
Point3D = Tuple[float, float, float]

class AllowArbitraryTypes:
    # TODO look into numpy.typing.NDArray
    # https://numpy.org/devdocs/reference/typing.html#numpy.typing.NDArray
    arbitrary_types_allowed = True

@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=False)
class Pose:
    position: np.ndarray  # shape: (3, )
    orientation: np.ndarray  # shape: (4, ), quaternion

    def __hash__(self) -> int:
        return hash((*self.position.tolist(), *self.orientation.tolist()))

    @validator("position")
    @classmethod
    def position_shape(cls, v: np.ndarray):
        if v.shape != (3,):
            raise ValueError("position must be 3D")
        return v

    def __post_init__(self):
        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]
        self.qx = self.orientation[0]
        self.qy = self.orientation[1]
        self.qz = self.orientation[2]
        self.qw = self.orientation[3]

    @property 
    def pos_string(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    @validator("orientation")
    @classmethod
    def orientation_shape(cls, v: np.ndarray):
        if v.shape != (4,):
            raise ValueError("orientation must be a 4D quaternion")
        return v

    @property
    def array(self) -> np.ndarray:
        return np.concatenate([self.position, self.orientation])
        
    @property
    def flattened(self) -> List[float]:
        return list(self.position) + list(self.orientation)

    def __eq__(self, other) -> bool:
        return bool(
            np.allclose(self.position, other.position)
            and np.allclose(self.orientation, other.orientation)
        )

    @property
    def matrix(self) -> np.ndarray:
        return affines.compose(
            T=self.position, R=quaternions.quat2mat(self.orientation), Z=np.ones(3)
        )

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose:
        T, R = affines.decompose(matrix)[:2]
        return Pose(position=T.copy(), orientation=quaternions.mat2quat(R.copy()))

    def transform(self, transform_matrix: np.ndarray) -> Pose:
        assert transform_matrix.shape == (
            4,
            4,
        ), f"expected 4x4 transformation matrix but got {transform_matrix.shape}"
        T, R, _, _ = affines.decompose(transform_matrix @ self.matrix)
        return Pose(position=T, orientation=quaternions.mat2quat(R))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        pos_str = ",".join(f"{x:.02f}" for x in self.position)
        rot_str = ",".join(f"{x:.02f}" for x in euler.quat2euler(self.orientation))
        return f"Pose(pos=({pos_str}),rot=({rot_str}))"

    def distance(self, other: Pose, orientation_factor: float = 0.05) -> float:
        position_distance = float(np.linalg.norm(self.position - other.position))

        orientation_distance = (
            float(
                quaternions.qnorm(
                    quaternions.qmult(
                        self.orientation,
                        quaternions.qinverse(other.orientation),
                    )
                )
            )
            if not np.allclose(
                euler.quat2euler(other.orientation),
                euler.quat2euler(self.orientation),
                rtol=0.1,
                atol=0.1,
            )
            else 0.0
        )
        dist = position_distance + orientation_factor * orientation_distance
        return dist


Pixel = Tuple[int, int]
Point3D = Tuple[float, float, float]

GRADIENT_COLORS = [
    "BuGn",
    "hot",
    "cool_r",
    "viridis",    
    "cividis", 
    "CMRmap",
    "gist_earth",
]
def visualize_voxel_scene(
    obs_pcd: PointCloud, 
    voxel_size: float = 0.02, 
    path_pts: List = [], 
    path_colors: List = [],
    save_img = False,
    img_path = 'test.jpg',
    expand_path = False,
    ):
    """ Displays the scene and path points as voxels """
    pcd = obs_pcd.to_open3d()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    new_voxels = []
    new_colors = []
    if len(path_colors) == 0:
        _colors = []
        for i, path in enumerate(path_pts):
            pal = sns.color_palette(GRADIENT_COLORS[i], len(path))
            _colors.append([np.array(color) for color in pal]) 
        path_colors = _colors
    else:
        assert len(path_pts) == len(path_colors), "Must have same number of paths and colors"
        
    for i, path in enumerate(path_pts):
        assert len(path[0]) >= 3, "Path points must be 3D or 7D"
        for j, pt in enumerate(path):
            new_voxels.append(pt[:3])
            color = path_colors[i][j] 
            if max(color) > 1:
                color = color/255
            new_colors.append(color)
            if expand_path:
                # add the 26 neighbors
                for dx, dy, dz in product([-1, 0, 1], repeat=3):
                    if dx == dy == dz == 0:
                        continue
                    new_voxels.append(pt[:3] + np.array([dx, dy, dz])*voxel_size)
                    new_colors.append(color)

    # o3d.visualization.draw_geometries([voxel_grid])
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window() 
    visualizer.add_geometry(voxel_grid)
    if len(new_voxels) > 0:
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(new_voxels)
        pcd_new.colors = o3d.utility.Vector3dVector(new_colors)
        new_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_new, voxel_size=voxel_grid.voxel_size)
        visualizer.add_geometry(new_grid)

    visualizer.poll_events()
    visualizer.update_renderer()
    
    view_control = visualizer.get_view_control()
    view_control.set_front([1, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_lookat([0, 0, 0])
    visualizer.run()
        
    color = visualizer.capture_screen_float_buffer(False)
    color = np.asarray(color) 
    # save the captured frame as a .jpg image
    if save_img:
        o3d.io.write_image(img_path, o3d.geometry.Image((color * 255).astype('uint8')))

    # visualizer.capture_screen_image('test.jpg')
    visualizer.destroy_window() 
    return  

@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=False)
class PointCloud:
    rgb_pts: np.ndarray
    segmentation_pts: Dict[str, np.ndarray]
    xyz_pts: np.ndarray

    @validator("rgb_pts")
    @classmethod
    def rgb_dtype(cls, rgb_pts: np.ndarray):
        if (
            (rgb_pts.dtype in {np.float32, np.float64})
            and rgb_pts.max() < 1.0
            and rgb_pts.min() > 0.0
        ):
            rgb_pts = rgb_pts * 255
            return rgb_pts.astype(np.uint8)
        elif rgb_pts.dtype == np.uint8:
            return rgb_pts
        else:
            raise ValueError(f"`rgb_pts` in unexpected format: dtype {rgb_pts.dtype}")

    @validator("segmentation_pts")
    @classmethod
    def segmentation_pts_shape(cls, v: Dict[str, np.ndarray]):
        for pts in v.values():
            if len(pts.shape) > 2:
                raise ValueError(f"points.shape should N, but got {pts.shape}")
        return v

    @validator("xyz_pts")
    @classmethod
    def xyz_pts_shape(cls, v: np.ndarray):
        if len(v.shape) != 2 or v.shape[1] != 3:
            raise ValueError("points should be Nx3")
        return v

    @validator("xyz_pts")
    @classmethod
    def same_len(cls, v: np.ndarray, values, field, config):
        if "rgb_pts" in values and len(values["rgb_pts"]) != len(v):
            raise ValueError("`len(rgb_pts) != len(xyz_pts)`")
        if "segmentation_pts" in values and not all(
            len(pts) == len(v) for pts in values["segmentation_pts"].values()
        ):
            raise ValueError("`len(segmentation_pts) != len(xyz_pts)`")
        return v

    def __len__(self):
        return len(self.xyz_pts)

    def __add__(self, other: PointCloud):
        return PointCloud(
            xyz_pts=np.concatenate((self.xyz_pts, other.xyz_pts), axis=0),
            rgb_pts=np.concatenate((self.rgb_pts, other.rgb_pts), axis=0),
            segmentation_pts={
                k: np.concatenate(
                    (self.segmentation_pts[k], other.segmentation_pts[k]), axis=0
                )
                for k in self.segmentation_pts.keys()
            },
        )

    def to_open3d(self, color_palette: str = 'colorblind') -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz_pts)
        pcd.colors = o3d.utility.Vector3dVector(self.rgb_pts.astype(float)/255.0)
        return pcd

    def voxel_downsample(
        self, voxel_dim: float = 0.015, skip_segmentation: bool = True
    ) -> PointCloud:
        pcd = self.to_open3d()
        pcd = pcd.voxel_down_sample(voxel_dim)
        xyz_pts = np.array(pcd.points).astype(np.float32)
        return PointCloud(
                xyz_pts=xyz_pts,
                rgb_pts=(np.array(pcd.colors)*255.0).astype(np.uint8),
                segmentation_pts={},
            )

    @property
    def normals(self) -> np.ndarray:
        pcd = self.to_open3d()
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
        )
        return np.asarray(pcd.normals)

    def filter_bounds(self, bounds: Tuple[Point3D, Point3D]):
        in_bounds_mask = np.logical_and(
            (self.xyz_pts > np.array(bounds[0])).all(axis=1),
            (self.xyz_pts < np.array(bounds[1])).all(axis=1),
        )
        return PointCloud(
            xyz_pts=self.xyz_pts[in_bounds_mask],
            rgb_pts=self.rgb_pts[in_bounds_mask],
            segmentation_pts={
                k: self.segmentation_pts[k][in_bounds_mask]
                for k in self.segmentation_pts.keys()
            },
        )

    def subsample(
        self, num_pts: int, numpy_random: np.random.RandomState
    ) -> PointCloud:
        indices = numpy_random.choice(
            len(self), size=num_pts, replace=num_pts > len(self)
        )
        return PointCloud(
            xyz_pts=self.xyz_pts[indices],
            rgb_pts=self.rgb_pts[indices],
            segmentation_pts={k: v[indices] for k, v in self.segmentation_pts.items()},
        )

    def __getitem__(self, key: str) -> PointCloud:
        assert key in self.segmentation_pts
        return PointCloud(
            xyz_pts=self.xyz_pts[self.segmentation_pts[key]],
            rgb_pts=self.rgb_pts[self.segmentation_pts[key]],
            segmentation_pts={
                key: np.ones(self.segmentation_pts[key].sum(), dtype=np.bool)
            },
        )

    def show(
        self: PointCloud,
        # background_color: Point3D = (0.1, 0.1, 0.1),
        background_color: Point3D = (1.0, 1.0, 1.0),
        views: Optional[Sequence[Tuple[float, float]]] = None,
        pts_size: float = 3,
        bounds: Optional[Tuple[Point3D, Point3D]] = ((-1.0, -0.7, 0.12), (1.0, 2.5, 1.5)),
        show: bool = True,
        img_output_name: Optional[str] = "point_cloud.png",
    ):
        if views is None:
            views = [(45, 135)]
        fig = plt.figure(figsize=(6, 6), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        point_cloud = self
        if bounds is not None:
            point_cloud = point_cloud.filter_bounds(bounds=bounds)
        x, y, z = (
            point_cloud.xyz_pts[:, 0],
            point_cloud.xyz_pts[:, 1],
            point_cloud.xyz_pts[:, 2],
        )
        ax.set_facecolor(background_color)
        ax.w_xaxis.set_pane_color(background_color)  # type: ignore
        ax.w_yaxis.set_pane_color(background_color)  # type: ignore
        ax.w_zaxis.set_pane_color(background_color)  # type: ignore

        ax.scatter(
            x,
            y,
            z,
            c=point_cloud.rgb_pts.astype(float)/255.0,
            s=pts_size,  # type: ignore
        )

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])  # type: ignore
        if bounds is not None:
            ax.axes.set_xlim3d(left=bounds[0][0], right=bounds[1][0])  # type: ignore
            ax.axes.set_ylim3d(bottom=bounds[0][1], top=bounds[1][1])  # type: ignore
            ax.axes.set_zlim3d(bottom=bounds[0][2], top=bounds[1][2])  # type: ignore
        plt.tight_layout(pad=0)
        # imgs = list(set_view_and_save_img(fig, ax, views))
        if show:
            plt.show()
        else:
            # save fig
            plt.savefig(img_output_name)
        plt.close(fig)
        return # imgs


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=False)
class VisionSensorOutput:
    rgb: np.ndarray
    depth: np.ndarray 
    pos: Point3D
    rot_mat: np.ndarray
    fov: float
    segmentation: Optional[Dict[str, np.ndarray]] = None

    @validator("depth")
    @classmethod
    def depth_shape(cls, v: np.ndarray):
        if v is not None and len(v.shape) != 2:
            raise ValueError("depth images should be HxW")
        return v

    @validator("rgb")
    @classmethod
    def rgb_shape(cls, v: np.ndarray):
        if v.dtype == np.float32:
            if v.min() < 0.0 or v.max() > 1.0:
                raise ValueError(
                    f"rgb values out of expected range: {v.min()},{v.max()}"
                )
            v = (v * 255).astype(np.uint8)
        if v.shape[-1] == 3:
            v = np.concatenate((v, np.ones_like(v[:, :, 0])[:, :, None] * 255), axis=2)
        if v.shape[-1] != 4:
            raise ValueError("rgb images should be HxWx4")
        return v

    @validator("rot_mat")
    @classmethod
    def check_rot_mat(cls, v: np.ndarray):
        if v.shape != (3, 3):
            raise ValueError("rot_mat matrix has incorrect shape")
        return v
    
    def __post_init__(self):
        # NOTE(Mandi): instead of setting it as property, this allows us to compute pointcloud only once 
        cloud = self.get_pointcloud()
        setattr(self, 'point_cloud', cloud)
        camera_matrix = self.get_camera_matrix()
        setattr(self, 'camera_matrix', camera_matrix)

    def get_pointcloud(self) -> PointCloud:
        img_h = self.depth.shape[0]
        img_w = self.depth.shape[1]

        # Project depth into 3D pointcloud in camera coordinates
        pixel_x, pixel_y = np.meshgrid(
            np.linspace(0, img_w - 1, img_w), np.linspace(0, img_h - 1, img_h)
        )
        cam_pts_x = np.multiply(
            pixel_x - self.intrinsic[0, 2], self.depth / self.intrinsic[0, 0]
        )
        cam_pts_y = np.multiply(
            pixel_y - self.intrinsic[1, 2], self.depth / self.intrinsic[1, 1]
        )
        cam_pts_z = self.depth
        cam_pts = (
            np.array([cam_pts_x, cam_pts_y, cam_pts_z])
            .transpose(1, 2, 0)
            .reshape(-1, 3)
        )
        world_pts = np.matmul(
            self.extrinsic,
            np.concatenate((cam_pts, np.ones_like(cam_pts[:, [0]])), axis=1).T,
        ).T[:, :3]
        if self.segmentation is not None:
            seg_pts = {k: v.reshape(-1) for k, v in self.segmentation.items()}
        else:
            seg_pts = {}
        cloud = PointCloud(
            xyz_pts=world_pts.astype(np.float32),
            rgb_pts=self.rgb[:, :, :3].reshape(-1, 3),
            segmentation_pts=seg_pts,
            # bboxes={k: np.zeros((2,3), dtype=float) for k in self.segmentation.keys()},
        )
        return cloud
        
    @property
    def valid_points_mask(self):
        assert (self.depth != 0).reshape(-1).all(), "not all depth values are valid"
        return np.ones_like((self.depth != 0).reshape(-1))

    @property
    def height(self) -> int:
        return self.depth.shape[0]

    @property
    def width(self) -> int:
        return self.depth.shape[1]

    @property
    def intrinsic(self) -> np.ndarray:
        if self.height != self.width:
            raise NotImplementedError("compute unequal focal lengths")
        focal_length = (self.height / 2) / np.tan(np.deg2rad(self.fov) / 2)
        return np.array(
            [
                [focal_length, 0, self.height / 2],
                [0, focal_length, self.width / 2],
                [0, 0, 1],
            ],
            dtype=float,
        )

    @property
    def forward(self) -> np.ndarray:
        forward = np.matmul(self.rot_mat, np.array([0, 0, -1]))
        return forward / np.linalg.norm(forward)

    @property
    def up(self) -> np.ndarray:
        up = np.matmul(self.rot_mat, np.array([0, 1, 0]))
        return up / np.linalg.norm(up)

    @property
    def extrinsic(self) -> np.ndarray:
        pos = self.pos
        forward = self.forward.copy()
        u = self.up.copy()
        s = np.cross(forward, u)
        s = s / np.linalg.norm(s)
        u = np.cross(s, forward)
        view_matrix = [
            s[0],            u[0],            -forward[0],          0,
            s[1],            u[1],            -forward[1],          0,
            s[2],            u[2],            -forward[2],          0,
            -np.dot(s, pos), -np.dot(u, pos), np.dot(forward, pos), 1,
        ]
        view_matrix = np.array(view_matrix).reshape(4, 4).T
        pose_matrix = np.linalg.inv(view_matrix)
        pose_matrix[:, 1:3] = -pose_matrix[:, 1:3]
        return pose_matrix

    def get_camera_matrix(self):
        # credit: mujoco tutorial code 
        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -np.array(self.pos)

        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = self.rot_mat.T

        # Focal transformation matrix (3x4).
        focal_scaling = (1.0 / np.tan(np.deg2rad(self.fov) / 2)) * self.height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (self.width - 1) / 2.0
        image[1, 2] = (self.height - 1) / 2.0
        return image @ focal @ rotation @ translation
 