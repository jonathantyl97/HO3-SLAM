"""
Stub heavy runtime dependencies so unit tests can import the sort modules
without needing a GPU, PyTorch, ROS, or OpenCV installation.

Strategy: make torch.nn.Module a real Python class so nn.Module subclasses
defined in deep_sort model code can be parsed without a real torch install.
"""
import sys
import types
from unittest.mock import MagicMock


def _make_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# ── PyTorch ───────────────────────────────────────────────────────────────────
# torch.nn.Module must be a real class for "class Foo(nn.Module)" to work.
class _FakeModule:
    def __init__(self, *a, **kw):
        pass
    def __call__(self, *a, **kw):
        return MagicMock()

class _FakeParameter:
    pass

if "torch" not in sys.modules:
    _torch = _make_module("torch")
    _torch.no_grad = lambda: MagicMock()
    _torch.Tensor = MagicMock
    _torch.load = MagicMock()
    _torch.cuda = MagicMock()
    _torch.device = MagicMock()

if "torch.nn" not in sys.modules:
    _nn = _make_module("torch.nn")
    _nn.Module = _FakeModule
    _nn.Sequential = _FakeModule
    _nn.Linear = _FakeModule
    _nn.BatchNorm2d = _FakeModule
    _nn.BatchNorm1d = _FakeModule
    _nn.ReLU = _FakeModule
    _nn.Conv2d = _FakeModule
    _nn.Dropout = _FakeModule
    _nn.AdaptiveAvgPool2d = _FakeModule
    _nn.MaxPool2d = _FakeModule
    _nn.AvgPool2d = _FakeModule
    _nn.Softmax = _FakeModule
    _nn.Parameter = _FakeParameter
    _nn.functional = MagicMock()

if "torch.nn.functional" not in sys.modules:
    sys.modules["torch.nn.functional"] = MagicMock()

for _name in ["torchvision", "torchvision.transforms",
              "torchvision.models", "torchvision.datasets"]:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()

# ── OpenCV ────────────────────────────────────────────────────────────────────
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

# ── ROS / sensor_msgs ─────────────────────────────────────────────────────────
for _name in [
    "rospy",
    "sensor_msgs",
    "sensor_msgs.msg",
    "sensor_msgs.point_cloud2",
    "geometry_msgs",
    "geometry_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
]:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()

_sensor_msg = sys.modules["sensor_msgs.msg"]
_sensor_msg.PointCloud2 = MagicMock
_sensor_msg.PointField = MagicMock()
_sensor_msg.PointField.FLOAT32 = 7
_sensor_msg.PointField.UINT32 = 6
