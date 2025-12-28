# src/motion_retargeting/motion_retargeting/retarget/__init__.py

def __getattr__(name):
    if name in ("BVHRetarget", "Joint"):
        from .retarget import BVHRetarget, Joint
        return {"BVHRetarget": BVHRetarget, "Joint": Joint}[name]
    raise AttributeError(name)

__all__ = ["BVHRetarget", "Joint"]
