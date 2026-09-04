import os
import sys


# 通常带中日韩字形的系统字体（按顺序尝试）。
_FONT_CANDIDATES = (
  "/System/Library/Fonts/PingFang.ttc",
  "/System/Library/Fonts/Hiragino Sans GB.ttc",
  "/System/Library/Fonts/STHeiti Light.ttc",
  "/Library/Fonts/Arial Unicode.ttf",
  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
  "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
  "C:/Windows/Fonts/msyh.ttc",
  "C:/Windows/Fonts/simhei.ttf",
)


def load_ui_font(loader):
  """返回支持中日韩的 DynamicTextFont；找不到则返回 None，改用 Panda 默认字体。"""
  for path in _FONT_CANDIDATES:
    if not os.path.isfile(path):
      continue
    font = loader.loadFont(path)
    if font is not None:
      return font

  if sys.platform == "darwin":
    fonts_dir = "/System/Library/Fonts"
    if os.path.isdir(fonts_dir):
      for name in sorted(os.listdir(fonts_dir)):
        if not name.endswith((".ttf", ".ttc", ".otf")):
          continue
        if not any(k in name for k in ("PingFang", "Heiti", "Hiragino", "Song")):
          continue
        path = os.path.join(fonts_dir, name)
        font = loader.loadFont(path)
        if font is not None:
          return font

  return None
