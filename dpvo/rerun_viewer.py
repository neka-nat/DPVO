import numpy as np
import torch


class RerunViewer:
    """Light-weight visualization bridge built on top of rerun."""

    def __init__(self, slam, app_id="DPVO", entity_path="world", max_points=200_000, lock_views=True):
        try:
            import rerun as rr
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The rerun viewer backend was selected, but the `rerun` package is "
                "not installed. Install it with `pip install rerun-sdk` and try again."
            ) from exc

        self.rr = rr
        self.slam = slam
        self.entity_path = entity_path.rstrip("/")
        self.max_points = max_points
        self.last_camera_index = -1
        self.lock_views = lock_views

        rr.init(app_id, spawn=True)
        self._setup_blueprint()

    def close(self):
        """Mirror DPViewer API."""

    def join(self):
        """Mirror DPViewer API."""

    def update_image(self, image: torch.Tensor):
        rr = self.rr
        rr.set_time_sequence("frame", int(self.slam.counter))

        np_image = (
            image.detach()
            .permute(1, 2, 0)
            .contiguous()
            .to(device="cpu", non_blocking=True)
            .numpy()
        )
        np_image = np_image[..., ::-1]  # BGR -> RGB
        rr.log(f"{self.entity_path}/camera/image", rr.Image(np_image))

        self._log_camera()
        self._log_trajectory()
        # self._log_points()

    def _setup_blueprint(self):
        if not self.lock_views:
            return

        try:
            import rerun.blueprint as rrbp
        except Exception:
            return

        image_path = f"{self.entity_path}/camera/image"
        try:
            self.rr.send_blueprint(
                rrbp.Blueprint(
                    rrbp.Horizontal(
                        rrbp.Spatial3DView(origin=self.entity_path, name="Scene 3D"),
                        rrbp.Spatial2DView(origin=image_path, name="Camera Image"),
                    )
                )
            )
        except Exception:
            pass

    def _log_camera(self):
        if self.slam.n == 0:
            return

        idx = self.slam.n - 1
        if idx == self.last_camera_index:
            return

        pose = (
            self.slam.pg.poses_[idx]
            .detach()
            .to(device="cpu", non_blocking=True)
            .numpy()
        )
        intrinsics = (
            self.slam.pg.intrinsics_[idx]
            .detach()
            .to(device="cpu", non_blocking=True)
            .numpy()
            * self.slam.RES
        )

        translation = pose[:3]
        quat_xyzw = pose[3:]

        self.rr.log(
            f"{self.entity_path}/camera",
            self.rr.Transform3D(
                translation=translation.tolist(),
                rotation=self.rr.Quaternion(xyzw=quat_xyzw.tolist()),
            ),
        )
        self.rr.log(
            f"{self.entity_path}/camera",
            self.rr.Pinhole(
                focal_length=intrinsics[:2].tolist(),
                principal_point=intrinsics[2:].tolist(),
                resolution=[int(self.slam.wd), int(self.slam.ht)],
            ),
            static=True,
        )

        self.last_camera_index = idx

    def _log_trajectory(self):
        if self.slam.n < 2:
            return

        poses = (
            self.slam.pg.poses_[: self.slam.n, :3]
            .detach()
            .to(device="cpu", non_blocking=True)
            .numpy()
        )
        self.rr.log(
            f"{self.entity_path}/trajectory",
            self.rr.LineStrips3D([poses.astype(np.float32)]),
        )

    def _log_points(self):
        if self.slam.m == 0:
            return

        points = self.slam.pg.points_[: self.slam.m]
        colors = self.slam.pg.colors_.view(-1, 3)[: self.slam.m]

        if points.shape[0] > self.max_points:
            idx = torch.linspace(
                0, points.shape[0] - 1, steps=self.max_points, dtype=torch.long, device=points.device
            )
            points = torch.index_select(points, 0, idx)
            colors = torch.index_select(colors, 0, idx)

        np_points = points.detach().to(device="cpu", non_blocking=True).numpy()
        np_colors = colors.detach().to(device="cpu", non_blocking=True).numpy()

        self.rr.log(
            f"{self.entity_path}/points",
            self.rr.Points3D(positions=np_points, colors=np_colors.astype(np.uint8)),
        )
