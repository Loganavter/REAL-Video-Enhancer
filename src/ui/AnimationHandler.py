from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect
from PySide6.QtCore import (
    QPropertyAnimation,
    QEasingCurve,
    QRect,
    QParallelAnimationGroup,
)


class AnimationHandler:
    """Page fades.

    A `QGraphicsOpacityEffect` on the page stack makes Qt render the whole stack into an offscreen surface and blit it.
    The effect used to stay installed after the animation finished: with it, the blit is not invalidated by a page
    switch, so the freshly shown page stayed BLACK (or stale) until some input forced a repaint - observed 2026-11-22 as
    "the interface appears only when you click it". The effect is now removed the moment the fade ends, and the stack is
    asked to repaint so the page paints directly from that point on.
    """

    def _fade(self, qObject: QWidget, start: float, end: float, on_done=None) -> None:
        effect = QGraphicsOpacityEffect(qObject)
        qObject.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(200)
        animation.setStartValue(start)
        animation.setEndValue(end)

        def _finish() -> None:
            # The effect must not outlive the fade: it changes how every child below is painted, and a page switch under
            # it leaves a stale blit. Removing it also drops the offscreen surface.
            qObject.setGraphicsEffect(None)
            qObject.update()
            if on_done is not None:
                on_done()

        animation.finished.connect(_finish)
        # Keep both alive: a QPropertyAnimation with no owner is garbage-collected mid-flight and the effect stays.
        self._animation = animation
        self._effect = effect
        animation.start()

    def fadeInAnimation(self, qObject: QWidget, n=None):
        self._fade(qObject, 0.0, 1.0)

    def fadeOutAnimation(self, qObject: QWidget, n=None):
        self._fade(qObject, 1.0, 0.0)

    def fade_to_color(self, color):
        self.animation = QPropertyAnimation(self, b"styleSheet")
        self.animation.setDuration(5000)  # Duration in milliseconds
        self.animation.setStartValue(self.styleSheet())
        self.animation.setEndValue(color)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()

        self.animation.start()

    def dropDownAnimation(self, widget: QWidget, duration=200):
        return
        start_geometry = widget.geometry()
        end_geometry = QRect(
            start_geometry.x(),
            start_geometry.y(),
            start_geometry.width(),
            start_geometry.height(),
        )

        # Move the widget above its final position
        start_geometry.moveTop(start_geometry.y() - start_geometry.height())

        self.animation = QPropertyAnimation(widget, b"geometry")
        self.animation.setDuration(duration)
        self.animation.setStartValue(start_geometry)
        self.animation.setEndValue(end_geometry)
        self.animation.setEasingCurve(QEasingCurve.Linear)
        self.animation.start()

    def moveUpAnimation(self, widget: QWidget, duration=200):
        return
        widget.setVisible(True)
        start_geometry = widget.geometry()
        end_geometry = QRect(
            start_geometry.x(),
            start_geometry.y() - start_geometry.height(),
            start_geometry.width(),
            start_geometry.height(),
        )

        start_geometry.moveTop(start_geometry.y() + start_geometry.height())

        self.animation = QPropertyAnimation(widget, b"geometry")
        self.animation.setDuration(duration)
        self.animation.setStartValue(start_geometry)
        self.animation.setEndValue(end_geometry)
        self.animation.setEasingCurve(QEasingCurve.Linear)
        self.animation.start()

    def dropDownFadeInAnimation(self, widget: QWidget, duration=500):
        return
        # Ensure the widget is initially hidden

        # Drop-down animation
        start_geometry = widget.geometry()
        end_geometry = QRect(
            start_geometry.x(),
            start_geometry.y(),
            start_geometry.width(),
            start_geometry.height(),
        )
        start_geometry.moveTop(start_geometry.y() - start_geometry.height())

        drop_down_animation = QPropertyAnimation(widget, b"geometry")
        drop_down_animation.setDuration(duration)
        drop_down_animation.setStartValue(start_geometry)
        drop_down_animation.setEndValue(end_geometry)
        drop_down_animation.setEasingCurve(QEasingCurve.OutBounce)

        # Fade-in animation
        opacity_effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(opacity_effect)
        fade_in_animation = QPropertyAnimation(opacity_effect, b"opacity")
        fade_in_animation.setDuration(duration)
        fade_in_animation.setStartValue(0)
        fade_in_animation.setEndValue(1)

        # Combine animations
        animation_group = QParallelAnimationGroup()
        animation_group.addAnimation(drop_down_animation)
        animation_group.addAnimation(fade_in_animation)

        # Show the widget and start the animation
        widget.setVisible(True)
        animation_group.start()
