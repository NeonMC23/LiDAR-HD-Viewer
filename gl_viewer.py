"""Legacy OpenGL fallback viewer used for debugging and compatibility."""
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_POINTS,
    glBegin,
    glClear,
    glColor3f,
    glEnable,
    glEnd,
    glLoadIdentity,
    glMatrixMode,
    glPointSize,
    glRotatef,
    glTranslatef,
    glVertex3f,
    glViewport,
    GL_MODELVIEW,
    GL_PROJECTION,
)
from OpenGL.GLU import gluPerspective


class GLViewer(QOpenGLWidget):
    """Legacy OpenGL point cloud widget kept for fallback/debug usage."""

    def __init__(self):
        super().__init__()
        self.points = None
        self.colors = None
        self.rot_x = -30.0
        self.rot_y = 45.0
        self.zoom = -50.0
        self.last_pos = None

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glPointSize(2.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w / max(1, h), 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, float(self.zoom))
        glRotatef(float(self.rot_x), 1.0, 0.0, 0.0)
        glRotatef(float(self.rot_y), 0.0, 1.0, 0.0)

        if self.points is None or len(self.points) == 0:
            return

        glBegin(GL_POINTS)
        for i, (x, y, z) in enumerate(self.points):
            if self.colors is not None and i < len(self.colors):
                c = self.colors[i]
                glColor3f(float(c[0]), float(c[1]), float(c[2]))
            else:
                glColor3f(1.0, 1.0, 1.0)
            glVertex3f(float(x), float(y), float(z))
        glEnd()

    def set_points(self, points, colors=None):
        self.points = points
        self.colors = colors
        self.update()

    def reset_view(self):
        self.rot_x = -30.0
        self.rot_y = 45.0
        self.zoom = -50.0
        self.update()

    def mousePressEvent(self, e):
        self.last_pos = e.position()

    def mouseMoveEvent(self, e):
        if self.last_pos is None:
            return
        dx = e.position().x() - self.last_pos.x()
        dy = e.position().y() - self.last_pos.y()
        self.rot_x += dy * 0.5
        self.rot_y += dx * 0.5
        self.last_pos = e.position()
        self.update()

    def mouseReleaseEvent(self, _e):
        self.last_pos = None

    def wheelEvent(self, e):
        self.zoom += e.angleDelta().y() * 0.01
        self.zoom = max(-20000.0, min(-1.0, self.zoom))
        self.update()

