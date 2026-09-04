"""离屏抓取策略观测（车头前视）与窗口截图。"""

from __future__ import annotations

import os
import sys

import numpy as np
from panda3d.core import (
  Camera,
  FrameBufferProperties,
  GraphicsEngine,
  GraphicsOutput,
  GraphicsPipe,
  GraphicsPipeSelection,
  NodePath,
  PerspectiveLens,
  Texture,
  WindowProperties,
  loadPrcFileData,
)
from PIL import Image

from drive_env.camera import EGO_FOV_DEG, EgoCamera


def enable_headless_prc() -> None:
  """须在创建离屏缓冲之前调用；开窗口路径不要走这里。"""
  loadPrcFileData("", "window-type none")
  loadPrcFileData("", "audio-library-name null")
  loadPrcFileData("", "notify-level-audio error")
  loadPrcFileData("", "sync-video #f")
  # 无可用 X11 时不要让 pandagl 抢默认管线（GPU 机器常带着失效的 DISPLAY=:0.0）。
  if sys.platform.startswith("linux") and not _linux_x11_usable():
    loadPrcFileData("", "load-display p3headlessgl")
    loadPrcFileData("", "aux-display p3tinydisplay")


def _linux_x11_usable() -> bool:
  """``DISPLAY`` 已设且能连上 X server 时为 True。"""
  display = os.environ.get("DISPLAY")
  if not display:
    return False
  import ctypes

  x11 = None
  for lib in ("libX11.so.6", "libX11.so"):
    try:
      x11 = ctypes.CDLL(lib)
      break
    except OSError:
      continue
  if x11 is None:
    return False
  x11.XOpenDisplay.restype = ctypes.c_void_p
  x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
  x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
  dpy = x11.XOpenDisplay(display.encode("utf-8"))
  if not dpy:
    return False
  x11.XCloseDisplay(dpy)
  return True


def _fb_prop_variants() -> list[FrameBufferProperties]:
  variants: list[FrameBufferProperties] = []
  for rgba, depth in (((8, 8, 8, 0), 16), ((8, 8, 8, 8), 24), ((8, 8, 8, 0), 24)):
    fb = FrameBufferProperties()
    fb.setRgbColor(True)
    fb.setRgbaBits(*rgba)
    fb.setDepthBits(depth)
    variants.append(fb)
  return variants


def _pipe_module_names() -> list[str | None]:
  """``None`` 表示 ``makeDefaultPipe()``。无 X11 的 Linux 优先 EGL。"""
  if sys.platform.startswith("linux") and not _linux_x11_usable():
    return ["p3headlessgl", "p3tinydisplay", None]
  return [None, "p3headlessgl", "p3tinydisplay"]


def _open_offscreen_buffer(
  engine: GraphicsEngine,
  name: str,
  width: int,
  height: int,
):
  """打开一块不依赖可见窗口的离屏缓冲。

  本机有显示器时仍走默认 OpenGL；Linux GPU 机器没有可用 X11 时改走
  ``p3headlessgl``（EGL），再不行用 ``p3tinydisplay`` 软件光栅。
  """
  selection = GraphicsPipeSelection.getGlobalPtr()
  win_props = WindowProperties.size(width, height)
  flag_sets = (
    GraphicsPipe.BFRefuseWindow,
    GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFFbPropsOptional,
  )
  tried: list[str] = []
  seen_pipes: set[int] = set()

  for module in _pipe_module_names():
    if module is None:
      pipe = selection.makeDefaultPipe()
      label = "default"
    else:
      pipe = selection.makeModulePipe(module)
      label = module
    if pipe is None:
      tried.append(f"{label}:no-pipe")
      continue
    pipe_id = id(pipe)
    if pipe_id in seen_pipes:
      continue
    seen_pipes.add(pipe_id)
    for fb_props in _fb_prop_variants():
      for flags in flag_sets:
        buffer = engine.makeOutput(
          pipe,
          name,
          -1,
          fb_props,
          win_props,
          flags,
        )
        if buffer is not None:
          if module is not None:
            print(f"headless capture: {label} {width}x{height}")
          return pipe, buffer
    tried.append(f"{label}:refused-buffer")

  display = os.environ.get("DISPLAY", "")
  raise RuntimeError(
    "graphics pipe refused an offscreen render buffer "
    f"(tried {', '.join(tried) or 'nothing'}; DISPLAY={display!r}). "
    "On Linux GPU boxes without X11, install EGL (libEGL + NVIDIA/Mesa GL) "
    "so p3headlessgl can load, or run under xvfb-run."
  )


def _texture_to_rgb_chw(tex: Texture, width: int, height: int) -> np.ndarray:
  data = memoryview(tex.getRamImage()).tobytes()
  h, w = tex.getYSize(), tex.getXSize()
  comps = tex.getNumComponents()
  arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, comps)
  rgb = np.flipud(arr[:, :, :3].copy())
  resized = Image.fromarray(rgb).resize((width, height), Image.BILINEAR)
  return np.transpose(np.array(resized, dtype=np.uint8), (2, 0, 1))


def capture_rgb_chw(show_base, width: int, height: int) -> np.ndarray:
  """渲染当前相机画面，返回 uint8 CHW 图像，形状 (3, H, W)。

  重要：不要在 ``win.getScreenshot()`` 之后再调用 ``Texture.makeRamImage()``。
  在 macOS/Cocoa 上那会重新分配一块空白 RAM 缓冲，得到全零图。
  ``getScreenshot()`` 已经提供了填好的 RAM 图像。
  """
  show_base.graphicsEngine.renderFrame()
  return _texture_to_rgb_chw(show_base.win.getScreenshot(), width, height)


class OffscreenCapture:
  """主相机的离屏渲染镜像，不含 2D 叠加层。

  从可见窗口抓帧需要隐藏 HUD 并强制多渲一帧，推理时窗口会闪一下。
  该缓冲与窗口每帧一起渲染，读取无副作用，返回的是上一帧画面。

  不要把这里的画面送给 PilotNet：窗口镜头是跟随相机，策略必须吃 ``EgoCapture``。
  """

  def __init__(self, show_base, width: int, height: int):
    self.width = width
    self.height = height

    fb_props = FrameBufferProperties()
    fb_props.setRgbColor(True)
    fb_props.setRgbaBits(8, 8, 8, 0)
    fb_props.setDepthBits(16)
    win_props = WindowProperties.size(
      show_base.win.getXSize(),
      show_base.win.getYSize(),
    )

    self.buffer = show_base.graphicsEngine.makeOutput(
      show_base.pipe,
      "model-view",
      -10,
      fb_props,
      win_props,
      GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFSizeTrackHost,
      show_base.win.getGsg(),
      show_base.win,
    )
    if self.buffer is None:
      raise RuntimeError("graphics pipe refused an offscreen render buffer")

    self.texture = Texture()
    self.buffer.addRenderTexture(self.texture, GraphicsOutput.RTMCopyRam)
    self.buffer.setClearColorActive(True)
    self.buffer.setClearColor(show_base.win.getClearColor())
    self.camera = show_base.makeCamera(
      self.buffer,
      lens=show_base.camLens,
      camName="model-cam",
    )

  @classmethod
  def create(cls, show_base, width: int, height: int) -> OffscreenCapture | None:
    try:
      return cls(show_base, width, height)
    except Exception as exc:  # noqa: BLE001 — 降级为窗口截图
      print(f"offscreen capture unavailable ({exc}); falling back to screenshots")
      return None

  def read_rgb_chw(self) -> np.ndarray | None:
    """返回上一帧已渲染画面；首次渲染前为 None。"""
    if not self.texture.hasRamImage():
      return None
    return _texture_to_rgb_chw(self.texture, self.width, self.height)


class EgoCapture:
  """策略观测用的离屏车头前视抓取器（采集 / PPO / eval / main 自动驾驶）。

  先以采集分辨率渲染再下采样到 PilotNet 分辨率。
  直接按 60×80 渲染会使路缘锯齿过重，BC 策略闭环成功率会塌掉。
  3D 窗口可以另挂 ChaseCamera，但不要把窗口画面送给网络。
  """

  def __init__(
    self,
    width: int,
    height: int,
    fov: float = EGO_FOV_DEG,
    render_width: int = 800,
    render_height: int = 600,
  ):
    self.width = width
    self.height = height
    self.render_width = render_width
    self.render_height = render_height
    self.engine = GraphicsEngine.getGlobalPtr()
    self.pipe, self.buffer = _open_offscreen_buffer(
      self.engine, "rl-ego", render_width, render_height
    )

    self.texture = Texture()
    self.buffer.addRenderTexture(self.texture, GraphicsOutput.RTMCopyRam)
    self.buffer.setClearColorActive(True)
    self.buffer.setClearColor((0.53, 0.75, 0.92, 1.0))

    self.lens = PerspectiveLens()
    self.lens.setFov(fov)
    self.lens.setNearFar(0.4, 500.0)
    self._dr = self.buffer.makeDisplayRegion()
    self.cam_np: NodePath | None = None
    self.ego: EgoCamera | None = None

  def bind(self, scene_root: NodePath, target: NodePath) -> None:
    """在 ``scene_root`` 下挂一台前视相机，对准 ``target`` 车头。"""
    if self.cam_np is not None:
      self.cam_np.removeNode()
    cam = Camera("policy-ego-cam")
    cam.setLens(self.lens)
    self.cam_np = scene_root.attachNewNode(cam)
    self._dr.setCamera(self.cam_np)
    self.ego = EgoCamera(self.cam_np, target)

  def sync_ego(self, dt: float = 0.0) -> None:
    if self.ego is None:
      raise RuntimeError("EgoCapture.bind() was not called")
    self.ego.update(dt)

  def read_rgb_chw(self, dt: float = 0.0, *, render: bool = True) -> np.ndarray:
    """更新前视姿态，返回 uint8 CHW (3, H, W)。

    ``render=True``（采集 / PPO / eval）：立刻 ``renderFrame``。
    ``render=False``（已有 ShowBase 窗口）：只读上一帧，避免任务里再渲一次。
    """
    self.sync_ego(dt)
    if render or not self.texture.hasRamImage():
      self.engine.renderFrame()
    if not self.texture.hasRamImage():
      self.engine.renderFrame()
    if not self.texture.hasRamImage():
      raise RuntimeError("offscreen ego capture produced no RAM image")
    return _texture_to_rgb_chw(self.texture, self.width, self.height)

  def close(self) -> None:
    if self.cam_np is not None:
      self.cam_np.removeNode()
      self.cam_np = None
    self.ego = None
    if self.buffer is not None:
      self.engine.removeWindow(self.buffer)
      self.buffer = None
