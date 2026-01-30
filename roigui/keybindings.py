"""Keyboard shortcut bindings."""

from PySide6.QtCore import Qt

# Default keybindings: key -> action name
DEFAULT_BINDINGS = {
    # Tools
    Qt.Key_A: "tool_add",
    Qt.Key_E: "tool_erase",
    Qt.Key_L: "tool_lasso",
    Qt.Key_X: "tool_extend",
    Qt.Key_R: "tool_refine",
    Qt.Key_Escape: "tool_none",
    # Accept/Reject
    Qt.Key_Return: "accept",
    Qt.Key_Enter: "accept",
    Qt.Key_Backspace: "reject",
    # View modes
    Qt.Key_1: "view_mean",
    Qt.Key_2: "view_correlation",
    Qt.Key_3: "view_local_correlation",
    # ROI management
    Qt.Key_N: "new_roi",
    Qt.Key_P: "propose_roi",
    Qt.Key_Tab: "next_roi",
    # Display
    Qt.Key_S: "toggle_show_all",
    Qt.Key_F: "toggle_fill",
    # Pen size
    Qt.Key_BracketLeft: "pen_smaller",
    Qt.Key_BracketRight: "pen_larger",
}
