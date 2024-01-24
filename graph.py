import math
import sys

from PyQt6.QtCore import QPointF, QTimer
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget


class Circle:
    def __init__(self, center, radius, attraction=0.1):
        self.center = center
        self.radius = radius
        self.attraction = attraction
        self.attract_pairs = []

    def add_attract_pair(self, circle):
        self.attract_pairs.append(circle)


class CircleDrawer(QWidget):
    def __init__(self):
        super().__init__()
        self.circles = [
            Circle(QPointF(100.0, 100.0), 50, 0.05),
            Circle(QPointF(200.0, 200.0), 50, 0.1),
            Circle(QPointF(300.0, 100.0), 50, 0.08),
            Circle(QPointF(150.0, 300.0), 50, 0.09),
            Circle(QPointF(250.0, 300.0), 50, 0.11),
        ]
        self.selected_circle = None
        self.offset = QPointF()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_physics)
        self.timer.start(10)  # Обновляем физику каждые 10 миллисекунд
        self.setMouseTracking(True)  # Разрешаем отслеживание мыши

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(255, 0, 0))  # Задаем цвет кругов (красный)

        for circle in self.circles:
            painter.setBrush(QColor(255, 0, 0))  # Заливаем цветом круга (красный)
            painter.drawEllipse(
                circle.center, circle.radius, circle.radius
            )  # Рисуем каждый из кругов

    def handle_collision(self, circle1, circle2):
        direction = circle2.center - circle1.center
        distance = math.sqrt(
            (circle1.center.x() - circle2.center.x()) ** 2
            + (circle1.center.y() - circle2.center.y()) ** 2
        )
        if distance < 1:
            distance = 1
        force = (circle1.radius + circle2.radius - distance) / distance

        direction *= force
        circle1.center -= direction / 2
        circle2.center += direction / 2

    def update_physics(self):
        for circle1 in self.circles:
            for circle2 in self.circles:
                if circle1 != circle2:
                    if circle2 in circle1.attract_pairs:
                        direction = circle2.center - circle1.center
                        distance = math.sqrt(
                            (circle1.center.x() - circle2.center.x()) ** 2
                            + (circle1.center.y() - circle2.center.y()) ** 2
                        )
                        if distance < 1:
                            distance = 1
                        force = (circle1.attraction * circle2.attraction) / distance

                        direction *= force
                        circle1.center += direction / 2
                        circle2.center -= direction / 2
                    elif self.check_collision(circle1, circle2):
                        self.handle_collision(circle1, circle2)

        self.update()  # Перерисовываем виджет при обновлении физики

    def check_collision(self, circle1, circle2):
        distance = math.sqrt(
            (circle1.center.x() - circle2.center.x()) ** 2
            + (circle1.center.y() - circle2.center.y()) ** 2
        )
        min_distance = circle1.radius + circle2.radius
        return distance < min_distance

    def mouseMoveEvent(self, event):
        if self.selected_circle:
            self.selected_circle.center = event.position() - self.offset

    def mousePressEvent(self, event):
        for circle in self.circles:
            if (
                math.sqrt(
                    (event.position().x() - circle.center.x()) ** 2
                    + (event.position().y() - circle.center.y()) ** 2
                )
                < circle.radius
            ):
                self.selected_circle = circle
                self.offset = event.position() - circle.center
                break

    def mouseReleaseEvent(self, event):
        self.selected_circle = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Перемещаемые круги с физикой")
        self.setGeometry(100, 100, 400, 400)

        self.circle_drawer = CircleDrawer()
        self.setCentralWidget(self.circle_drawer)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
