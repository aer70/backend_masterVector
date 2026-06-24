"""
Точная векторизация BMP-схем в SVG без искажений геометрии.

Основные принципы:
- Перебор всех уникальных цветов (RGB, учитывая альфу) и построение бинарной маски для каждого цвета.
- По маске строятся контуры через OpenCV (findContours, режим CCOMP) — так сохраняются внешние границы и отверстия.
- Для каждого внешнего контура и его дыр формируется единый path с fill-rule=evenodd. Координаты — в пикселях без масштабирования.
- Никаких эвристик (Hough, «распознавание прямоугольников», сглаживание линий) — максимум точности.
- OCR (pytesseract) опционален и выключен по умолчанию, чтобы не влиять на геометрию.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import cv2
import numpy as np
import svgwrite
from PIL import Image

try:
    import pytesseract  # type: ignore
    _TESS = True
except Exception:
    pytesseract = None  # type: ignore
    _TESS = False

from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename


# -----------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------

def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def is_black(rgb: tuple[int, int, int], threshold: int = 0) -> bool:
    """Возвращает True, если цвет близок к чёрному. threshold — максимальное значение канала.
    Пример: threshold=8 — всё, где R,G,B <= 8, считается чёрным.
    """
    r, g, b = rgb
    return r <= threshold and g <= threshold and b <= threshold


def get_background_color(img_rgba: np.ndarray) -> tuple[int, int, int]:
    """Определяет цвет фона по границам изображения (мода).

    img_rgba: HxWx4 (uint8)
    Возвращает RGB-кортеж.
    """
    h, w, _ = img_rgba.shape
    # Собираем пиксели границы, игнорируем полностью прозрачные
    border = []
    # Верх/низ
    top = img_rgba[0, :, :]
    bottom = img_rgba[h - 1, :, :]
    # Лево/право
    left = img_rgba[:, 0, :]
    right = img_rgba[:, w - 1, :]
    for row in (top, bottom):
        mask = row[:, 3] > 0
        if np.any(mask):
            border.extend([tuple(px[:3]) for px in row[mask]])
    for col in (left, right):
        mask = col[:, 3] > 0
        if np.any(mask):
            border.extend([tuple(px[:3]) for px in col[mask]])
    if not border:
        # На всякий случай — если вся граница прозрачная, берём моду по всему изображению
        rgb = img_rgba[:, :, :3].reshape(-1, 3)
        return tuple(map(int, Counter(map(tuple, rgb)).most_common(1)[0][0]))
    return Counter(border).most_common(1)[0][0]


def contours_to_evenodd_path(
    mask: np.ndarray,
    simplify_epsilon: float = 0.0,
    *,
    enable_smoothing: bool = True,
    # Снап 90° углов (замена пары 45° поворотов одной вершиной на внешнем угле пикселя)
    snap_90deg_corners: bool = True,
    snap_90_one_pixel_only: bool = True,
    # Новый алгоритм лесенки: удалить все внутренние вершины в сериях чередующихся осевых ходов (H,V,H,V,...) 
    stair2_remove_interior: bool = True,
    stair2_min_len: int = 3,
    # Новый симметричный алгоритм: удалять внешние вершины в сериях HVHV...
    stair2_remove_exterior: bool = True,
    stair2_exterior_min_len: int = 3,
    stair2_exterior_only_outer: bool = True,
    # Параметры stair3 (выполняется после stair4)
    stair3_delete_by_first: bool = True,
    stair3_min_edges: int = 4,
    stair3_max_step_px: float | None = 7.0,
    # Политика для stair3 (не используется, т.к. stair3 отключён)
    stair3_policy: str = "by_depth",
    simplify_collinear: bool = True,
    collinear_tol: float = 0.0,
    # stair4: удаление внутренних тупых углов (~135°). Перед этим локально удаляем коллинеарные узлы
    stair4_enable: bool = True,
    stair4_target_angle_deg: float = 135.0,   # целевой угол
    stair4_angle_tol_deg: float = 15.0,       # погрешность ±15° => диапазон 120°..150°
    stair4_only_internal: bool = True,        # удалять только внутренние (вогнутые) вершины
    stair4_min_polygon_vertices: int = 5,     # применять только если после чистки >= N вершин
    # 8) ra90: анализ правых углов и удаление по правилам серий HV с короткими рёбрами
    ra90_enable: bool = True,
    ra90_max_edge_len_px: float = 6.0,
    ra90_min_edges: int = 6,  # минимум узлов/вершин в последовательности = 6
    avoid_overlapping_vertices: bool = True,
    simplify_stair_triplets: bool = True,
    stair_max_distance_px: float | None = 6.0,
    stair_action: str = "average",  # "average" (усреднить p1..p3), "remove13" (удалить p1 и p3)
    stair_collapse_runs: bool = True,
    stair_run_min_len: int = 2,
    stair_run_keep_lowest_only: bool = True,
) -> list[tuple[str, tuple[int, int]]]:
    """Возвращает список path-d строк для каждого «внешнего контура + его отверстия».

    mask: бинарная маска uint8 {0, 255}
    simplify_epsilon: допуск аппроксимации в пикселях. 0 — без упрощения.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Добавляем 1px рамку вокруг маски, чтобы корректно обрабатывать объекты, касающиеся границ изображения
    mask_padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Контуры с иерархией: двухуровневый список (внешние и отверстия)
    contours, hierarchy = cv2.findContours(mask_padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]  # shape: (N, 4) [next, prev, child, parent]

    def approx(cnt: np.ndarray) -> np.ndarray:
        # Для максимальной точности по умолчанию не упрощаем.
        # Если epsilon > 0, применяем, но только для достаточно больших контуров.
        if simplify_epsilon and simplify_epsilon > 0:
            peri = cv2.arcLength(cnt, True)
            # Не упрощаем очень маленькие контуры (тонкие линии, 1-2 пикселя)
            if peri < 20:
                return cnt
            eps = max(0.0, float(simplify_epsilon))
            return cv2.approxPolyDP(cnt, eps, True)
        return cnt

    def remove_collinear(points: np.ndarray) -> np.ndarray:
        # Удаляет точки, лежащие на одной прямой (ровно противоположные направления соседних рёбер)
        # points: Nx2
        n = points.shape[0]
        if n <= 3:
            return points
        # Итеративно до стабилизации: проход и удаление коллинеарных
        pts = points.copy().tolist()
        changed = True
        tol = max(0.0, float(collinear_tol))
        while changed:
            changed = False
            m = len(pts)
            if m <= 3:
                break
            keep = [True] * m
            for i in range(m):
                prev = pts[(i - 1) % m]
                curr = pts[i]
                nxt = pts[(i + 1) % m]
                vx1 = curr[0] - prev[0]
                vy1 = curr[1] - prev[1]
                vx2 = nxt[0] - curr[0]
                vy2 = nxt[1] - curr[1]
                # Коллинеарность: векторное произведение ~ 0
                cross = vx1 * vy2 - vy1 * vx2
                if abs(cross) <= tol:
                    # И сонаправленность/противоположность: скалярное произведение < 0 для разворота, >=0 для продолжения
                    # Нам важно убирать любые коллинеарные промежуточные точки (продолжение одной прямой)
                    keep[i] = False
                    changed = True
            if changed:
                pts = [p for k, p in zip(keep, pts) if k]
        return np.array(pts, dtype=np.int32)

    def dedupe_points(points: np.ndarray) -> np.ndarray:
        # Удаляет подряд идущие одинаковые точки (перекрывающиеся вершины)
        if points.shape[0] <= 1:
            return points
        out = [points[0].tolist()]
        for i in range(1, points.shape[0]):
            if points[i, 0] != out[-1][0] or points[i, 1] != out[-1][1]:
                out.append(points[i].tolist())
        # если последняя совпала с первой (теоретически), можно не закрывать явно — у нас 'Z'
        return np.array(out, dtype=np.int32)

    def ring_to_d(cnt: np.ndarray, *, ring_depth: int) -> str:
        # cnt: Nx1x2; координаты в пикселях (x, y)
        points = cnt.reshape(-1, 2)
        if len(points) == 0:
            return ""
        if avoid_overlapping_vertices:
            points = dedupe_points(points)

    # Примечание: алгоритм stair3 отключён и в этой функции не используется

        # 90° corner snapping: Axis -> Diagonal -> Axis, две «синие» точки меняем на один узел в внешнем углу пикселя
        if snap_90deg_corners and len(points) >= 4:
            pts = points.copy().tolist()
            changed = True
            while changed and len(pts) >= 4:
                changed = False
                m = len(pts)
                for i in range(m):
                    j0 = i
                    j1 = (i + 1) % m
                    j2 = (i + 2) % m
                    j3 = (i + 3) % m
                    p0 = pts[j0]
                    p1 = pts[j1]
                    p2 = pts[j2]
                    p3 = pts[j3]

                    dx1 = int(np.sign(p1[0] - p0[0])); dy1 = int(np.sign(p1[1] - p0[1]))
                    dx2 = int(np.sign(p2[0] - p1[0])); dy2 = int(np.sign(p2[1] - p1[1]))
                    dx3 = int(np.sign(p3[0] - p2[0])); dy3 = int(np.sign(p3[1] - p2[1]))

                    is_axis1 = (dx1 == 0) ^ (dy1 == 0)
                    is_diag2 = (dx2 != 0 and dy2 != 0)
                    is_axis3 = (dx3 == 0) ^ (dy3 == 0)
                    if not (is_axis1 and is_diag2 and is_axis3):
                        continue
                    # Ортогональность осевых сегментов
                    if not ((dx1 == 0 and dy3 == 0) or (dy1 == 0 and dx3 == 0)):
                        continue
                    # Диагональ должна «смотреть» в ту же четверть
                    if dx1 == 0 and dy3 == 0:
                        # вертикаль -> диагональ -> горизонталь
                        if not (dx2 == dx3 and dy2 == dy1):
                            continue
                        if snap_90_one_pixel_only and (abs(p2[0] - p1[0]) != 1 or abs(p2[1] - p1[1]) != 1):
                            continue
                        new_pt = [p1[0], p2[1]]
                    else:
                        # горизонталь -> диагональ -> вертикаль
                        if not (dx2 == dx1 and dy2 == dy3):
                            continue
                        if snap_90_one_pixel_only and (abs(p2[0] - p1[0]) != 1 or abs(p2[1] - p1[1]) != 1):
                            continue
                        new_pt = [p2[0], p1[1]]

                    # Разворачиваем массив, если четверка пересекает начало, чтобы удалить линейно
                    if not (j0 < j1 < j2 < j3):
                        rot = j0
                        pts = pts[rot:] + pts[:rot]
                        m = len(pts)
                        j0, j1, j2, j3 = 0, 1, 2, 3
                    # Заменяем p1 на снапнутую вершину, удаляем p2 (две «синие» схлопываются в одну «белую»)
                    pts[j1] = new_pt
                    del pts[j2]
                    changed = True
                    break
            points = np.array(pts, dtype=np.int32)
            if avoid_overlapping_vertices and len(points) >= 2:
                points = dedupe_points(points)
        # Удаление/сглаживание «лесенок»
        if enable_smoothing and simplify_stair_triplets:
            pts = points.copy().tolist()
            changed = True
            thr2: float | None
            if stair_max_distance_px is None:
                thr2 = None
            else:
                t = float(stair_max_distance_px)
                thr2 = t * t
            # Вариант 1: коллапс серий повторяющихся лесенок, идущих в одном направлении (без «закруглений»)
            if stair_collapse_runs and len(pts) >= 4:
                m = len(pts)
                matches = []  # (i, (dx,dy), midType, pIdx[4])
                for i in range(m):
                    p0 = pts[i]
                    p1 = pts[(i+1) % m]
                    p2 = pts[(i+2) % m]
                    p3 = pts[(i+3) % m]
                    dx1 = np.sign(p1[0] - p0[0]); dy1 = np.sign(p1[1] - p0[1])
                    dx2 = np.sign(p2[0] - p1[0]); dy2 = np.sign(p2[1] - p1[1])
                    dx3 = np.sign(p3[0] - p2[0]); dy3 = np.sign(p3[1] - p2[1])
                    is_diag1 = (dx1 != 0 and dy1 != 0)
                    if not (is_diag1 and dx3 == dx1 and dy3 == dy1):
                        continue
                    midType = None
                    if dx2 == dx1 and dy2 == 0:
                        midType = "H"
                    elif dx2 == 0 and dy2 == dy1:
                        midType = "V"
                    if midType is None:
                        continue
                    # дистанционный порог
                    if thr2 is not None:
                        dx03 = p3[0] - p0[0]
                        dy03 = p3[1] - p0[1]
                        if (dx03 * dx03 + dy03 * dy03) > thr2:
                            continue
                    matches.append((i, (int(dx1), int(dy1)), midType, [i, (i+1)%m, (i+2)%m, (i+3)%m]))

                # Группировка в серии по одинаковой диагонали и соседним индексам
                runs: list[list[tuple]] = []
                if matches:
                    matches.sort(key=lambda x: x[0])
                    current = [matches[0]]
                    for a, b in zip(matches, matches[1:]):
                        i_a, dir_a, mt_a, _ = a
                        i_b, dir_b, mt_b, _ = b
                        if dir_a == dir_b and (i_b == i_a + 1):
                            current.append(b)
                        else:
                            runs.append(current)
                            current = [b]
                    runs.append(current)

                # Маркируем индексы для удаления, оставляя в нижней тройке самую нижнюю точку
                to_delete: set[int] = set()
                for run in runs:
                    if len(run) < max(1, int(stair_run_min_len)):
                        continue
                    # Найти «самую нижнюю» тройку среди run (максимальный y среди её трёх p1..p3)
                    def triplet_low_key(entry: tuple) -> int:
                        _, _, _, idxs = entry
                        j1, j2, j3 = idxs[1], idxs[2], idxs[3]
                        return max(pts[j1][1], pts[j2][1], pts[j3][1])

                    lowest_entry = max(run, key=triplet_low_key)
                    lowest_idxs = lowest_entry[3]
                    lj1, lj2, lj3 = lowest_idxs[1], lowest_idxs[2], lowest_idxs[3]
                    # В самой нижней тройке оставить самую нижнюю точку
                    keep_idx = lj1
                    if pts[lj2][1] >= pts[keep_idx][1]:
                        keep_idx = lj2
                    if pts[lj3][1] >= pts[keep_idx][1]:
                        keep_idx = lj3

                    for entry in run:
                        _, _, _, idxs = entry
                        j1, j2, j3 = idxs[1], idxs[2], idxs[3]
                        if stair_run_keep_lowest_only and entry is lowest_entry:
                            # Удаляем две из трёх, кроме самой нижней
                            for jj in (j1, j2, j3):
                                if jj != keep_idx:
                                    to_delete.add(jj)
                        else:
                            # Удаляем всю тройку (p1,p2,p3)
                            to_delete.update([j1, j2, j3])

                if to_delete:
                    # Строим новый список точек, пропуская помеченные
                    new_pts = [p for idx, p in enumerate(pts) if idx not in to_delete]
                    # Защита: замкнутый контур требует >=3
                    if len(new_pts) >= 3:
                        pts = new_pts
                        changed = True
                # Если не было изменений, можно попробовать локальную обработку по старой схеме
            # Вариант 2: локальная обработка одиночных/редких лесенок
            while changed and len(pts) >= 4:
                changed = False
                m = len(pts)
                for i in range(m):
                    p0 = pts[i]
                    p1 = pts[(i+1) % m]
                    p2 = pts[(i+2) % m]
                    p3 = pts[(i+3) % m]
                    dx1 = np.sign(p1[0] - p0[0]); dy1 = np.sign(p1[1] - p0[1])
                    dx2 = np.sign(p2[0] - p1[0]); dy2 = np.sign(p2[1] - p1[1])
                    dx3 = np.sign(p3[0] - p2[0]); dy3 = np.sign(p3[1] - p2[1])
                    # Ищем обобщённый шаблон «лесенка» в 8 направлениях:
                    # Диагональ (sx,sy) с sx,sy ∈ {-1,1}, затем осевой шаг, совпадающий по знаку с одной из компонент:
                    # (sx,sy), (sx,0) или (0,sy), (sx,sy)
                    is_diag1 = (dx1 != 0 and dy1 != 0)
                    matches = False
                    if is_diag1 and dx3 == dx1 and dy3 == dy1:
                        # Горизонтальный середний шаг в сторону dx1
                        if dx2 == dx1 and dy2 == 0:
                            matches = True
                        # Вертикальный середний шаг в сторону dy1
                        elif dx2 == 0 and dy2 == dy1:
                            matches = True
                    if matches:
                        # Доп. условие: расстояние между крайними точками шаблона не превышает порог
                        if thr2 is not None:
                            dx03 = p3[0] - p0[0]
                            dy03 = p3[1] - p0[1]
                            if (dx03 * dx03 + dy03 * dy03) > thr2:
                                continue
                        # Действие над тройкой точек
                        action = str(stair_action).lower().strip()
                        if action == "remove13":
                            # Удаляем p1 и p3 (сохраняем p2)
                            j1 = (i + 1) % m
                            j3 = (i + 3) % m
                            del pts[j3]
                            m = len(pts)
                            if j3 < j1:
                                j1 -= 1
                            del pts[j1]
                        else:
                            # Усредняем p1, p2, p3 между собой, p0 не трогаем
                            j1 = (i + 1) % m
                            j2 = (i + 2) % m
                            j3 = (i + 3) % m
                            mx = int(round((pts[j1][0] + pts[j2][0] + pts[j3][0]) / 3.0))
                            my = int(round((pts[j1][1] + pts[j2][1] + pts[j3][1]) / 3.0))
                            pts[j1] = [mx, my]
                            pts[j2] = [mx, my]
                            pts[j3] = [mx, my]
                        changed = True
                        break
            points = np.array(pts, dtype=np.int32)
        # 4) Новый алгоритм: чередующиеся осевые повороты (лесенка)
        # 4a) Удалить внутренние углы в сериях HVHV
        if stair2_remove_interior and len(points) >= 4:
            pts = points.copy().tolist()

            def poly_area2(ps: list[list[int]] | list[tuple[int,int]]):
                s = 0
                m = len(ps)
                for i in range(m):
                    x1, y1 = ps[i]
                    x2, y2 = ps[(i + 1) % m]
                    s += x1 * y2 - x2 * y1
                return s  # вдвое площадь, знак = ориентация (CCW > 0)

            def edge_kind(a: list[int], b: list[int]) -> str:
                dx = int(np.sign(b[0] - a[0])); dy = int(np.sign(b[1] - a[1]))
                if dx == 0 and dy != 0:
                    return 'V'
                if dy == 0 and dx != 0:
                    return 'H'
                return 'D'

            n = len(pts)
            orient = 1 if poly_area2(pts) > 0 else -1  # +1 для CCW, -1 для CW
            kinds = [edge_kind(pts[i], pts[(i + 1) % n]) for i in range(n)]

            visited: set[int] = set()
            to_delete: set[int] = set()

            def turn_sign(i_prev: int, i_curr: int, i_next: int) -> int:
                x1, y1 = pts[i_curr][0] - pts[i_prev][0], pts[i_curr][1] - pts[i_prev][1]
                x2, y2 = pts[i_next][0] - pts[i_curr][0], pts[i_next][1] - pts[i_curr][1]
                cross = x1 * y2 - y1 * x2
                if cross > 0:
                    return 1
                if cross < 0:
                    return -1
                return 0

            i = 0
            min_edges = max(3, int(stair2_min_len))  # минимум рёбер в серии (HVH) = 3
            while i < n:
                if i in visited:
                    i += 1
                    continue
                k0 = kinds[i]
                k1 = kinds[(i + 1) % n]
                if (k0 in ('H','V')) and (k1 in ('H','V')) and (k0 != k1):
                    # старт серии
                    run_edges = [i]
                    j = i + 1
                    while True:
                        ek_prev = kinds[(j - 1) % n]
                        ek = kinds[j % n]
                        if (ek in ('H','V')) and (ek != ek_prev):
                            run_edges.append(j % n)
                            j += 1
                            if len(run_edges) > n:
                                break
                        else:
                            break
                    for e in run_edges:
                        visited.add(e)
                    if len(run_edges) >= min_edges:
                        # Вершины серии: p_{edge+1}. Удаляем те, где поворот = «внутрь» фигуры
                        for t in range(len(run_edges) - 1):
                            v_idx = (run_edges[t] + 1) % n
                            i_prev = (v_idx - 1) % n
                            i_next = (v_idx + 1) % n
                            sgn = turn_sign(i_prev, v_idx, i_next)
                            interior_sgn = 1 if orient > 0 else -1
                            if sgn == interior_sgn:
                                to_delete.add(v_idx)
                    i = j
                else:
                    i += 1

            if to_delete:
                pts2 = [p for idx, p in enumerate(pts) if idx not in to_delete]
                if len(pts2) >= 3:
                    pts = pts2
                    points = np.array(pts, dtype=np.int32)
                    if avoid_overlapping_vertices:
                        points = dedupe_points(points)

        # 4b) Удалить внешние углы в сериях HVHV (как на правом примере),
        #     по умолчанию — только на внешних контурах (depth чётная)
        if stair2_remove_exterior and len(points) >= 4:
            if (not stair2_exterior_only_outer) or ((ring_depth % 2) == 0):
                pts = points.copy().tolist()

                def poly_area2x(ps: list[list[int]] | list[tuple[int,int]]):
                    s = 0
                    m = len(ps)
                    for i in range(m):
                        x1, y1 = ps[i]
                        x2, y2 = ps[(i + 1) % m]
                        s += x1 * y2 - x2 * y1
                    return s

                def edge_kindx(a: list[int], b: list[int]) -> str:
                    dx = int(np.sign(b[0] - a[0])); dy = int(np.sign(b[1] - a[1]))
                    if dx == 0 and dy != 0:
                        return 'V'
                    if dy == 0 and dx != 0:
                        return 'H'
                    return 'D'

                n = len(pts)
                orient = 1 if poly_area2x(pts) > 0 else -1
                kinds = [edge_kindx(pts[i], pts[(i + 1) % n]) for i in range(n)]
                visited: set[int] = set()
                to_delete: set[int] = set()

                def turn_signx(i_prev: int, i_curr: int, i_next: int) -> int:
                    x1, y1 = pts[i_curr][0] - pts[i_prev][0], pts[i_curr][1] - pts[i_prev][1]
                    x2, y2 = pts[i_next][0] - pts[i_curr][0], pts[i_next][1] - pts[i_curr][1]
                    c = x1 * y2 - y1 * x2
                    if c > 0: return 1
                    if c < 0: return -1
                    return 0

                i = 0
                min_edges = max(3, int(stair2_exterior_min_len))
                while i < n:
                    if i in visited:
                        i += 1
                        continue
                    k0 = kinds[i]
                    k1 = kinds[(i + 1) % n]
                    if (k0 in ('H','V')) and (k1 in ('H','V')) and (k0 != k1):
                        run_edges = [i]
                        j = i + 1
                        while True:
                            ek_prev = kinds[(j - 1) % n]
                            ek = kinds[j % n]
                            if (ek in ('H','V')) and (ek != ek_prev):
                                run_edges.append(j % n)
                                j += 1
                                if len(run_edges) > n:
                                    break
                            else:
                                break
                        for e in run_edges:
                            visited.add(e)
                        if len(run_edges) >= min_edges:
                            for t in range(len(run_edges) - 1):
                                v_idx = (run_edges[t] + 1) % n
                                i_prev = (v_idx - 1) % n
                                i_next = (v_idx + 1) % n
                                sgn = turn_signx(i_prev, v_idx, i_next)
                                interior_sgn = 1 if orient > 0 else -1
                                # удаляем внешние повороты
                                if sgn == -interior_sgn:
                                    to_delete.add(v_idx)
                        i = j
                    else:
                        i += 1

                if to_delete:
                    pts2 = [p for idx, p in enumerate(pts) if idx not in to_delete]
                    if len(pts2) >= 3:
                        pts = pts2
                        points = np.array(pts, dtype=np.int32)
                        if avoid_overlapping_vertices:
                            points = dedupe_points(points)
    # 5) (устар.) — ранее здесь находился stair3; теперь он выполняется позже, после stair4

        # 6) stair4: удалить внутренние вершины с тупым углом ~135° (после локального удаления коллинеарных)
        if stair4_enable and len(points) >= max(4, int(stair4_min_polygon_vertices)):
            # Локально уберём коллинеарные точки, чтобы угол считался корректно
            pts4 = remove_collinear(points)
            if avoid_overlapping_vertices:
                pts4 = dedupe_points(pts4)
            if pts4.shape[0] >= max(4, int(stair4_min_polygon_vertices)):
                target = float(stair4_target_angle_deg)
                tol = float(stair4_angle_tol_deg)
                # Предрасчёт косинусов границ допуска
                import math
                c_lo = math.cos(math.radians(min(179.9, target + tol)))
                c_hi = math.cos(math.radians(max(0.1, target - tol)))
                # Определим ориентацию полигона для внутреннего/внешнего знака поворота
                def area2(pa: np.ndarray) -> int:
                    s = 0
                    m = pa.shape[0]
                    for i in range(m):
                        x1, y1 = int(pa[i,0]), int(pa[i,1])
                        x2, y2 = int(pa[(i+1)%m,0]), int(pa[(i+1)%m,1])
                        s += x1*y2 - x2*y1
                    return s
                orient = 1 if area2(pts4) > 0 else -1  # +1 = CCW
                m = pts4.shape[0]
                keep = [True]*m
                for i in range(m):
                    p_prev = pts4[(i-1)%m].astype(int)
                    p_cur  = pts4[i].astype(int)
                    p_next = pts4[(i+1)%m].astype(int)
                    v1 = np.array([p_prev[0]-p_cur[0], p_prev[1]-p_cur[1]], dtype=float)
                    v2 = np.array([p_next[0]-p_cur[0], p_next[1]-p_cur[1]], dtype=float)
                    n1 = np.hypot(v1[0], v1[1]); n2 = np.hypot(v2[0], v2[1])
                    if n1 == 0 or n2 == 0:
                        continue
                    v1 /= n1; v2 /= n2
                    cos_a = float(v1[0]*v2[0] + v1[1]*v2[1])  # cos внутреннего угла
                    # Внутренний vs внешний: используем знак векторного произведения
                    cross = (p_prev[0]-p_cur[0])*(p_next[1]-p_cur[1]) - (p_prev[1]-p_cur[1])*(p_next[0]-p_cur[0])
                    turn_sgn = 1 if cross > 0 else (-1 if cross < 0 else 0)
                    is_internal = (turn_sgn == (1 if orient>0 else -1))
                    # cos 135° ≈ -0.7071. Ищем cos в [c_hi, c_lo] с учётом убывания cos на [0..180]
                    match_angle = (cos_a <= c_hi) and (cos_a >= c_lo)
                    if match_angle and ((not stair4_only_internal) or is_internal):
                        keep[i] = False
                pts4_new = np.array([p for k,p in zip(keep, pts4) if k], dtype=np.int32)
                if pts4_new.shape[0] >= 3:
                    points = pts4_new
                    if avoid_overlapping_vertices:
                        points = dedupe_points(points)

        # 7) stair3: удалить вершины серии HVHV по типу, противоположному типу стартовой вершины
        # Выполняется после stair4
        if stair3_delete_by_first and len(points) >= 5:
            pts = points.copy().tolist()

            # Повернём список, чтобы сначала шла вершина с минимальным (y, x)
            def find_top_left_index(ps: list[list[int]] | list[tuple[int,int]]):
                best_i = 0
                best_y = ps[0][1]
                best_x = ps[0][0]
                for idx, (x, y) in enumerate(ps[1:], start=1):
                    if y < best_y or (y == best_y and x < best_x):
                        best_y = y; best_x = x; best_i = idx
                return best_i

            def rotate(ps: list[list[int]], k: int) -> list[list[int]]:
                k = k % len(ps)
                return ps[k:] + ps[:k]

            s0 = find_top_left_index(pts)
            if s0 != 0:
                pts = rotate(pts, s0)
            # Предпочтительно начать с горизонтального ребра вправо
            m0 = len(pts)
            chose = 0
            for t in range(m0):
                x1, y1 = pts[t]
                x2, y2 = pts[(t + 1) % m0]
                if (y2 == y1) and (x2 > x1):  # вправо
                    chose = t
                    break
            if chose != 0:
                pts = rotate(pts, chose)

            def poly_area2b(ps: list[list[int]] | list[tuple[int,int]]):
                s = 0
                m = len(ps)
                for i in range(m):
                    x1, y1 = ps[i]
                    x2, y2 = ps[(i + 1) % m]
                    s += x1 * y2 - x2 * y1
                return s

            def edge_kind2(a: list[int], b: list[int]) -> str:
                dx = int(np.sign(b[0] - a[0])); dy = int(np.sign(b[1] - a[1]))
                if dx == 0 and dy != 0:
                    return 'V'
                if dy == 0 and dx != 0:
                    return 'H'
                return 'D'

            def turn_sign_idx(idx_prev: int, idx_curr: int, idx_next: int) -> int:
                x1 = pts[idx_curr][0] - pts[idx_prev][0]
                y1 = pts[idx_curr][1] - pts[idx_prev][1]
                x2 = pts[idx_next][0] - pts[idx_curr][0]
                y2 = pts[idx_next][1] - pts[idx_curr][1]
                c = x1 * y2 - y1 * x2
                return 1 if c > 0 else (-1 if c < 0 else 0)

            n = len(pts)
            orient = 1 if poly_area2b(pts) > 0 else -1
            to_delete3: set[int] = set()
            min_edges = max(4, int(stair3_min_edges))
            step_thr2 = None if stair3_max_step_px is None else float(stair3_max_step_px) * float(stair3_max_step_px)

            # Два полных прохода по контуру
            for _pass in range(2):
                kinds = [edge_kind2(pts[i], pts[(i + 1) % n]) for i in range(n)]
                visited_edges: set[int] = set()
                i = 0
                while i < n:
                    if i in visited_edges:
                        i += 1
                        continue
                    k0 = kinds[i]
                    if k0 not in ('H','V'):
                        i += 1
                        continue
                    # Порог длины текущего ребра (i -> i+1)
                    if step_thr2 is not None:
                        a0 = pts[i]
                        b0 = pts[(i + 1) % n]
                        dx0 = b0[0] - a0[0]
                        dy0 = b0[1] - a0[1]
                        if (dx0*dx0 + dy0*dy0) > step_thr2:
                            i += 1
                            continue
                    run = [i]
                    j = i + 1
                    while True:
                        ek_prev = kinds[(j - 1) % n]
                        ek = kinds[j % n]
                        if (ek in ('H','V')) and (ek != ek_prev):
                            run.append(j % n)
                            j += 1
                            if len(run) > n:
                                break
                        else:
                            break
                    for e in run:
                        visited_edges.add(e)
                    if len(run) >= min_edges:
                        # Доп. проверка: внутренние шаги серии не длиннее порога
                        if step_thr2 is not None:
                            ok = True
                            for eidx in run[1:-1]:
                                a = pts[eidx]
                                b = pts[(eidx + 1) % n]
                                dx = b[0] - a[0]
                                dy = b[1] - a[1]
                                if (dx*dx + dy*dy) > step_thr2:
                                    ok = False
                                    break
                            if not ok:
                                i = j
                                continue
                        # Не ищем первую неколлинеарную вершину — используем sgn1 = 0
                        v_first = (run[0] + 1) % n
                        v_last_exclusive = (run[-1] + 1) % n
                        sgn1 = 0
                        interior_sgn = 1 if orient > 0 else -1
                        # Политика выбора типа удаляемых углов
                        policy = str(stair3_policy).lower().strip()
                        if policy == 'prefer_external':
                            target_sgn = -interior_sgn
                        elif policy == 'prefer_internal':
                            target_sgn = interior_sgn
                        elif policy == 'by_depth':
                            if (ring_depth % 2) == 0:
                                target_sgn = -interior_sgn
                            else:
                                target_sgn = interior_sgn
                        else:
                            # 'auto' — противоположно типу стартовой вершины; при sgn1=0 => interior_sgn
                            target_sgn = -interior_sgn if sgn1 == interior_sgn else interior_sgn
                        end_vertex = (run[-1] + 1) % n
                        curr = (run[0] + 1) % n
                        while curr != end_vertex:
                            s = turn_sign_idx((curr - 1) % n, curr, (curr + 1) % n)
                            if s == target_sgn:
                                to_delete3.add(curr)
                            curr = (curr + 1) % n
                    i = j

            if to_delete3:
                points = np.array([p for idx, p in enumerate(pts) if idx not in to_delete3], dtype=np.int32)
                if avoid_overlapping_vertices:
                    points = dedupe_points(points)

        # 8) ra90: последовательности прямых углов с короткими рёбрами
        # Правила:
        #  - серия определяется как чередование H/V с длиной каждого ребра <= ra90_max_edge_len_px
        #  - если серия начинается и заканчивается одинаковым типом угла (направлением), удаляем вершины противоположного типа
        #  - если начинается одним типом, а заканчивается другим — удаляем внешние углы контура
        #  - возможен старт серии не с прямого угла: серию всё равно выделяем по H/V чередованию
        if ra90_enable and len(points) >= 4:
            pts = points.copy().tolist()
            n = len(pts)
            import math
            thr2 = float(ra90_max_edge_len_px) * float(ra90_max_edge_len_px)
            def edge_kind(a: list[int], b: list[int]) -> str:
                dx = b[0] - a[0]; dy = b[1] - a[1]
                if dx == 0 and dy != 0:
                    return 'V'
                if dy == 0 and dx != 0:
                    return 'H'
                return 'D'
            def edge_len2(a: list[int], b: list[int]) -> int:
                dx = b[0] - a[0]; dy = b[1] - a[1]
                return dx*dx + dy*dy
            def area2(ps: list[list[int]]):
                s = 0
                m = len(ps)
                for i in range(m):
                    x1,y1=ps[i]; x2,y2=ps[(i+1)%m]
                    s += x1*y2 - x2*y1
                return s
            orient = 1 if area2(pts) > 0 else -1
            kinds = [edge_kind(pts[i], pts[(i+1)%n]) for i in range(n)]
            dels: set[int] = set()
            i = 0
            visited: set[int] = set()
            while i < n:
                if i in visited:
                    i += 1
                    continue
                k0 = kinds[i]
                if k0 not in ('H','V'):
                    i += 1
                    continue
                # серия по коротким рёбрам и HV чередованию
                run_edges = [i]
                j = i + 1
                ok_len = (edge_len2(pts[i], pts[(i+1)%n]) <= thr2)
                if not ok_len:
                    i += 1
                    continue
                while True:
                    ek_prev = kinds[(j-1)%n]
                    ek = kinds[j % n]
                    if (ek in ('H','V')) and (ek != ek_prev) and (edge_len2(pts[j % n], pts[(j+1)%n]) <= thr2):
                        run_edges.append(j % n)
                        j += 1
                        if len(run_edges) > n:
                            break
                    else:
                        break
                for e in run_edges:
                    visited.add(e)
                if len(run_edges) < max(3, int(ra90_min_edges)):
                    i = j
                    continue
                # Углы серии — вершины v = edge+1
                verts = [ (e+1) % n for e in run_edges ]
                # Классификация угла: направление (квадрант) и внутренний/внешний
                def turn_sign(idx_prev:int, idx_curr:int, idx_next:int) -> int:
                    x1 = pts[idx_curr][0]-pts[idx_prev][0]
                    y1 = pts[idx_curr][1]-pts[idx_prev][1]
                    x2 = pts[idx_next][0]-pts[idx_curr][0]
                    y2 = pts[idx_next][1]-pts[idx_curr][1]
                    c = x1*y2 - y1*x2
                    return 1 if c>0 else (-1 if c<0 else 0)
                def corner_dir(idx_prev:int, idx_curr:int, idx_next:int) -> tuple[int,int]:
                    # нормализованное направление «смотрит» угол (знак по x,y)
                    vx1 = np.sign(pts[idx_prev][0]-pts[idx_curr][0])
                    vy1 = np.sign(pts[idx_prev][1]-pts[idx_curr][1])
                    vx2 = np.sign(pts[idx_next][0]-pts[idx_curr][0])
                    vy2 = np.sign(pts[idx_next][1]-pts[idx_curr][1])
                    # для прямого угла — один из векторов осевой, другой тоже осевой; сумма даёт «квадрант» угла
                    return (int(vx1+vx2), int(vy1+vy2))
                sgn_interior = 1 if orient>0 else -1
                start_v = verts[0]; end_v = verts[-1]
                start_sgn = turn_sign((start_v-1)%n, start_v, (start_v+1)%n)
                end_sgn   = turn_sign((end_v-1)%n,   end_v,   (end_v+1)%n)
                start_dir = corner_dir((start_v-1)%n, start_v, (start_v+1)%n)
                end_dir   = corner_dir((end_v-1)%n,   end_v,   (end_v+1)%n)
                same_dir = (start_dir == end_dir)
                # Собираем список вершин для удаления по правилам
                to_del_run: list[int] = []
                if same_dir:
                    # Удаляем вершины противоположного типа (по направлению) относительно start_dir
                    opp = (-start_dir[0], -start_dir[1])
                    for v in verts:
                        d = corner_dir((v-1)%n, v, (v+1)%n)
                        if d == opp:
                            to_del_run.append(v)
                else:
                    # Удаляем внешние углы контура
                    for v in verts:
                        s = turn_sign((v-1)%n, v, (v+1)%n)
                        if s == -sgn_interior:
                            to_del_run.append(v)
                # К накопителю, удалим после полного прохода
                for v in to_del_run:
                    dels.add(v)
                i = j
            if dels:
                points = np.array([p for idx,p in enumerate(pts) if idx not in dels], dtype=np.int32)
                if avoid_overlapping_vertices:
                    points = dedupe_points(points)

        # Финальное удаление коллинеарных узлов: выполняется после всех видов сглаживания.
        if simplify_collinear:
            points = remove_collinear(points)
            if avoid_overlapping_vertices:
                points = dedupe_points(points)

        # Завершение функции ring_to_d
        if len(points) < 3:
            return ""
        # SVG path: M x y L x y ... Z
        # Вычитаем 1px смещение из-за добавленной рамки
        segs = [f"M{int(points[0,0]-1)},{int(points[0,1]-1)}"]
        for x, y in points[1:]:
            segs.append(f"L{int(x-1)},{int(y-1)}")
        segs.append("Z")
        return " ".join(segs)

    paths: list[tuple[str, tuple[int, int]]] = []

    def add_with_all_descendants(root_idx: int) -> None:
    # Жёсткий порядок: сначала полностью обрабатываем внешний контур,
    # затем все внутренние (по уровням, BFS), и только после этого переходим к следующему объекту.
        root_cnt = approx(contours[root_idx])
        d_parts = [ring_to_d(root_cnt, ring_depth=0)]

        # Соберём всех потомков в порядке уровней (BFS): сначала все дети,
        # затем их дети и т.д., чтобы не перемежать обработку разных веток.
        queue = []
        ci = hierarchy[root_idx][2]
        while ci != -1:
            queue.append(ci)
            ci = hierarchy[ci][0]

        q_idx = 0
        while q_idx < len(queue):
            node = queue[q_idx]
            q_idx += 1
            # Глубина кольца: считаем расстояние до корня по parent-цепочке
            depth = 0
            p = node
            while p != -1 and p != root_idx:
                p = hierarchy[p][3]
                depth += 1
            d_parts.append(ring_to_d(approx(contours[node]), ring_depth=depth))
            # добавим детей этого узла в конец очереди
            sub = hierarchy[node][2]
            while sub != -1:
                queue.append(sub)
                sub = hierarchy[sub][0]

        # Центроид по внешнему контуру; корректируем -1px рамку
        M = cv2.moments(root_cnt)
        if M['m00'] != 0:
            cx = int(round(M['m10'] / M['m00'])) - 1
            cy = int(round(M['m01'] / M['m00'])) - 1
        else:
            x, y, w_b, h_b = cv2.boundingRect(root_cnt)
            cx = x + w_b // 2 - 1
            cy = y + h_b // 2 - 1
        paths.append((" ".join(d_parts), (cx, cy)))

    # Проходим все контуры верхнего уровня (parent == -1)
    for idx, h in enumerate(hierarchy):
        if h[3] == -1:  # parent == -1
            add_with_all_descendants(idx)
    return paths


def bmp_to_svg_precise(
    input_bmp: str,
    output_svg: str,
    *,
    simplify_epsilon: float = 0.0,
    min_area: int = 1,
    ignore_background: bool = True,
    draw_background: bool = True,
    color_merge_step: int = 1,
    method: str = "rle",  # "rle" (идеальная копия) | "contours" (меньше узлов)
    pre_scale_factor: float = 1.0,  # Увеличить входное изображение, 2.0 => каждый пиксель станет 2x2 (NEAREST)
    group_objects: bool = True,
    min_group_area: int = 100,
    exclude_pipes_by_hsv: bool = False,
    pipe_hsv_low: tuple[int, int, int] = (80, 80, 30),
    pipe_hsv_high: tuple[int, int, int] = (140, 255, 255),
    thin_max_width: int = 3,
    thin_min_aspect: float = 4.0,
    bridge_gaps: bool = True,
    connect_same_color: bool = True,
    connect_radius_px: float = 1.0,
    connect_op: str = "close",  # "close" (мягко: мостит узкие щели) | "dilate" (жёстко: расширяет) | "none" (без изменений)
    connect_4neighbors: bool = False,  # True — 4-соседство (крест), False — 8-соседство (квадрат)
    enable_smoothing: bool = True,
    snap_90deg_corners: bool = True,
    snap_90_one_pixel_only: bool = True,
    stair2_remove_interior: bool = True,
    stair2_min_len: int = 3,
    stair2_remove_exterior: bool = True,
    stair2_exterior_min_len: int = 3,
    stair2_exterior_only_outer: bool = True,
    stair3_delete_by_first: bool = True,
    stair3_min_edges: int = 4,
    stair3_max_step_px: float | None = 7.0,
    stair3_policy: str = "by_depth",
    simplify_collinear: bool = True,
    collinear_tol: float = 0.0,
    # stair4: удалить внутренние тупые углы (~135°)
    stair4_enable: bool = True,
    stair4_target_angle_deg: float = 135.0,
    stair4_angle_tol_deg: float = 15.0,  # диапазон по умолчанию 120°..150°
    stair4_only_internal: bool = True,
    stair4_min_polygon_vertices: int = 5,
    # ra90 — анализ коротких прямых углов
    ra90_enable: bool = True,
    ra90_max_edge_len_px: float = 6.0,
    ra90_min_edges: int = 6,
    avoid_overlapping_vertices: bool = True,
    simplify_stair_triplets: bool = True,
    stair_max_distance_px: float | None = 6.0,
    stair_action: str = "average",
    stair_collapse_runs: bool = True,
    stair_run_min_len: int = 2,
    stair_run_keep_lowest_only: bool = True,
    bring_black_to_front: bool = True,
    black_threshold: int = 8,
    ocr: bool = False,
    tesseract_cmd: str | None = None,
) -> None:
    """Главная функция точной векторизации.

    simplify_epsilon: допуск упрощения контуров в пикселях (0 — без упрощения).
    min_area: пропускать компоненты меньше этой площади (в пикселях).
    ignore_background: исключать фон из векторизации.
    ocr: опционально добавить текстовые элементы (не влияет на геометрию фигур).
    tesseract_cmd: путь к tesseract.exe (Windows), если требуется.
    """
    # 1) Чтение изображения как RGBA без потери палитры
    try:
        image = Image.open(input_bmp).convert("RGBA")
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        return

    img = np.array(image)  # HxWx4, uint8
    # 1a) Опциональное увеличение разрешения (NEAREST)
    if pre_scale_factor is not None:
        sf = float(pre_scale_factor)
        if sf != 1.0 and sf > 0:
            h0, w0, _ = img.shape
            new_w = max(1, int(round(w0 * sf)))
            new_h = max(1, int(round(h0 * sf)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    h, w, _ = img.shape
    print(f"Изображение: {w}x{h}, dtype={img.dtype}")

    # 2) Определяем фон
    bg_rgb = get_background_color(img)
    print(f"Определён фон: {bg_rgb}")

    # 3) Опциональная мягкая группировка близких цветов (улучшает непрерывность тонких линий)
    if color_merge_step < 1:
        color_merge_step = 1
    if color_merge_step == 1:
        qrgb = img[:, :, :3]
    else:
        step = int(color_merge_step)
        # Округление к ближайшему значению с шагом step
        tmp = img[:, :, :3].astype(np.int16)
        qrgb = ((tmp + step // 2) // step * step).clip(0, 255).astype(np.uint8)

    # Используем квантованные RGB при построении групп, альфу берём из исходного
    qi = np.dstack((qrgb, img[:, :, 3:4]))
    flat = qi.reshape(-1, 4)
    unique_rgba, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]  # по убыванию площади
    unique_rgba = unique_rgba[order]
    counts = counts[order]
    print(f"Уникальных цветов (RGBA): {len(unique_rgba)}")

    # 4) Готовим SVG-документ
    dwg = svgwrite.Drawing(
        filename=output_svg,
        size=(w, h),
        profile="tiny",
        viewBox=f"0 0 {w} {h}",
    )
    # Кристально чёткие края для рендеров
    dwg.attribs['shape-rendering'] = 'crispEdges'
    dwg.set_desc(title="bmp2svg precise", desc="Автоматическая векторизация без искажений геометрии")

    # Рисуем фон как сплошной прямоугольник первым слоем
    if draw_background and bg_rgb is not None:
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill=rgb_to_hex(tuple(map(int, bg_rgb)))))

    # Коллекторы чёрных фигур
    black_pending = []  # для режима RLE — откладываем, чтобы позже вынести на передний план
    black_paths_pending = []  # чёрные path'ы (contours/connect), не попавшие в чёрные регионы

    # Построим «чёрные регионы» один раз по всему изображению и будем использовать для группировки
    def build_black_regions() -> list[dict]:
        regions: list[dict] = []
        thr = int(black_threshold)
        # Бинарная маска чёрных пикселей с учётом альфы
        black_mask = (
            (qrgb[:, :, 0] <= thr)
            & (qrgb[:, :, 1] <= thr)
            & (qrgb[:, :, 2] <= thr)
            & (img[:, :, 3] > 0)
        ).astype(np.uint8) * 255
        if np.count_nonzero(black_mask) == 0:
            return regions
        # Ищем внешние контуры «чёрных областей»
        cnts, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cnt is None or len(cnt) < 3:
                continue
            x, y, ww, hh = cv2.boundingRect(cnt)
            # Отсекаем регионы, касающиеся рамки изображения — вероятный фон/рамка
            if x <= 0 or y <= 0 or (x + ww) >= w or (y + hh) >= h:
                continue
            area = float(cv2.contourArea(cnt))
            if area < max(4.0, float(min_area)):
                continue
            # Сохраняем bbox и полигон; группу добавим позже в SVG (чтобы можно было вынести на передний план)
            bbox = (int(x), int(y), int(x + ww - 1), int(y + hh - 1))
            # Приводим к ожидаемой форме для pointPolygonTest (Nx1x2, float)
            poly = cnt.astype(np.float32)
            regions.append({
                'bbox': bbox,
                'poly': poly,
                # Разделяем на два слоя внутри региона: содержимое и чёрные контуры
                'content_group': dwg.g(),
                'black_group': dwg.g(),
                'has_content': False,
                'has_black': False,
            })
        return regions

    black_regions_data = build_black_regions()

    # 5) Группировка объектов по связности (необязательная)
    labels = None
    groups = {}
    if group_objects:
        # 5.1 Бинарная маска «не фон» — без учёта цветов (шаг 1: объединяем всё, что соприкасается)
        non_bg = (img[:, :, 3] > 0)
        if ignore_background and bg_rgb is not None:
            non_bg &= ~(
                (img[:, :, 0] == bg_rgb[0]) & (img[:, :, 1] == bg_rgb[1]) & (img[:, :, 2] == bg_rgb[2])
            )
        mask_u8 = (non_bg.astype(np.uint8) * 255)
        # 5.2 Замыкаем зазоры длиной 1 пиксель (3x3 close) и используем 8-связность
        if bridge_gaps:
            kernel = np.ones((3, 3), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        # 5.3 (Опционально) исключение линейных труб — можно включить позже, сейчас по умолчанию False
        if exclude_pipes_by_hsv:
            hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2HSV)
            low = np.array(pipe_hsv_low, dtype=np.uint8)
            high = np.array(pipe_hsv_high, dtype=np.uint8)
            pipe_cand = cv2.inRange(hsv, low, high) > 0
            if pipe_cand.any():
                cc_u8 = (pipe_cand.astype(np.uint8) * 255)
                nlab_p, lab_p = cv2.connectedComponents(cc_u8, connectivity=8)
                pipe_mask = np.zeros_like(pipe_cand)
                for i in range(1, nlab_p):
                    comp = (lab_p == i)
                    ys, xs = np.where(comp)
                    if ys.size == 0:
                        continue
                    minx, maxx = int(xs.min()), int(xs.max())
                    miny, maxy = int(ys.min()), int(ys.max())
                    w_box = maxx - minx + 1
                    h_box = maxy - miny + 1
                    min_dim = min(w_box, h_box)
                    aspect = max(w_box, h_box) / max(1, min_dim)
                    if min_dim <= int(thin_max_width) or aspect >= float(thin_min_aspect):
                        pipe_mask |= comp
                mask_u8 = mask_u8 & (~(pipe_mask.astype(np.uint8) * 255))
        # 5.4 Связные компоненты объектов
        nlabels, lab = cv2.connectedComponents(mask_u8, connectivity=8)
        if nlabels > 1 and int(min_group_area) > 1:
            areas = np.bincount(lab.flatten())
            keep_idxs = np.where(areas >= int(min_group_area))[0]
            keep = set(keep_idxs.tolist())
            keep.discard(0)
            lab = lab * np.isin(lab, list(keep))
        labels = lab.astype(np.int32)
        print(f"Групп найдено: {int(labels.max()) if labels is not None else 0}")

    def point_in_bbox(pt: tuple[int,int], bbox: tuple[int,int,int,int]) -> bool:
        x0, y0, x1, y1 = bbox
        x, y = pt
        return (x0 <= x <= x1) and (y0 <= y <= y1)

    def place_path_in_black_region(path_elem, centroid: tuple[int,int], color_rgb: tuple[int,int,int]) -> bool:
        """Пытается поместить path внутрь подходящего чёрного региона. Возвращает True, если помещён.
        Внутри региона чёрные пути добавляются в отдельный слой (black_group), чтобы быть поверх содержимого."""
        if not black_regions_data:
            return False
        for reg in black_regions_data:
            bbox = reg['bbox']
            if not point_in_bbox(centroid, bbox):
                continue
            # Точный тест точки внутри полигона
            inside = cv2.pointPolygonTest(reg['poly'], (float(centroid[0]), float(centroid[1])), False)
            if inside >= 0:  # внутри или на границе
                if is_black(color_rgb, black_threshold):
                    reg['black_group'].add(path_elem)
                    reg['has_black'] = True
                else:
                    reg['content_group'].add(path_elem)
                    reg['has_content'] = True
                return True
        return False

    def render_by_contours():
        processed = 0
        for (r, g, b, a), area in zip(unique_rgba, counts):
            if a == 0:
                continue  # прозрачное — пропускаем
            rgb = (int(r), int(g), int(b))
            if ignore_background and bg_rgb is not None and tuple(rgb) == tuple(bg_rgb):
                continue  # не дублируем фон, если он уже отрисован прямоугольником
            if area < max(1, int(min_area)):
                continue

            # Маска по квантованным каналам (сохраняет все близкие оттенки)
            match_rgb = (
                (qrgb[:, :, 0] == r)
                & (qrgb[:, :, 1] == g)
                & (qrgb[:, :, 2] == b)
                & (img[:, :, 3] > 0)
            )
            mask = np.where(match_rgb, 255, 0).astype(np.uint8)

            paths_d = contours_to_evenodd_path(
                mask,
                simplify_epsilon=simplify_epsilon,
                enable_smoothing=enable_smoothing,
                snap_90deg_corners=snap_90deg_corners,
                snap_90_one_pixel_only=snap_90_one_pixel_only,
                stair2_remove_interior=stair2_remove_interior,
                stair2_min_len=stair2_min_len,
                stair2_remove_exterior=stair2_remove_exterior,
                stair2_exterior_min_len=stair2_exterior_min_len,
                stair2_exterior_only_outer=stair2_exterior_only_outer,
                stair3_delete_by_first=stair3_delete_by_first,
                stair3_min_edges=stair3_min_edges,
                stair3_max_step_px=stair3_max_step_px,
                stair3_policy=stair3_policy,
                simplify_collinear=simplify_collinear,
                collinear_tol=collinear_tol,
                stair4_enable=stair4_enable,
                stair4_target_angle_deg=stair4_target_angle_deg,
                stair4_angle_tol_deg=stair4_angle_tol_deg,
                stair4_only_internal=stair4_only_internal,
                stair4_min_polygon_vertices=stair4_min_polygon_vertices,
                ra90_enable=ra90_enable,
                ra90_max_edge_len_px=ra90_max_edge_len_px,
                ra90_min_edges=ra90_min_edges,
                avoid_overlapping_vertices=avoid_overlapping_vertices,
                simplify_stair_triplets=simplify_stair_triplets,
                stair_max_distance_px=stair_max_distance_px,
                stair_action=stair_action,
                stair_collapse_runs=stair_collapse_runs,
                stair_run_min_len=stair_run_min_len,
                stair_run_keep_lowest_only=stair_run_keep_lowest_only,
            )
            if not paths_d:
                continue

            color_tuple = (int(r), int(g), int(b))
            fill = rgb_to_hex(color_tuple)
            for (d, centroid) in paths_d:
                path = dwg.path(d=d, fill=fill, stroke='none')
                path.attribs['fill-rule'] = 'evenodd'
                path.attribs['shape-rendering'] = 'crispEdges'
                if is_black(color_tuple, black_threshold):
                    # Добавляем чёрный путь внутрь подходящего заранее найденного чёрного региона
                    placed = place_path_in_black_region(path, centroid, color_tuple)
                    if not placed:
                        # Нет соответствующего региона — откладываем на верхний слой, чтобы его ничего не перекрыло
                        if bring_black_to_front:
                            black_paths_pending.append(path)
                        else:
                            dwg.add(path)
                else:
                    # Если попадает внутрь какого-нибудь чёрного региона — добавляем в его группу
                    if not place_path_in_black_region(path, centroid, color_tuple):
                        dwg.add(path)

            processed += 1
            if processed % 20 == 0:
                print(f"  Обработано цветов (contours): {processed}/{len(unique_rgba)}")

    def render_by_rle():
        # Пиксельно-точный режим: рисуем прямоугольники-«раны», объединяя одинаковые от строки к строке.
        alpha = img[:, :, 3] > 0
        open_rects = {}
        # ключ: (r,g,b, x, w) -> [x, y0, w, h]

        for y in range(h):
            row_rgb = qrgb[y]
            row_a = alpha[y]
            # Маскируем прозрачные как особый цвет, чтобы не попадали в ряды
            valid = row_a
            # Вычисляем границы ран по цвету
            # Строим массив ключей цвета с dtype u32 для векторного сравнения
            color_key = row_rgb[:, 0].astype(np.uint32) << 16 | row_rgb[:, 1].astype(np.uint32) << 8 | row_rgb[:, 2].astype(np.uint32)
            color_key = np.where(valid, color_key, 0xFFFFFFFF)
            # Индексы изменений
            diff = np.empty(w, dtype=bool)
            diff[0] = True
            diff[1:] = color_key[1:] != color_key[:-1]
            idx = np.where(diff)[0]
            # Границы отрезков
            starts = idx
            ends = np.r_[idx[1:], w]

            # Текущий набор открытых ключей для этой строки
            present_keys = set()

            for xs, xe in zip(starts, ends):
                ck = int(color_key[xs])
                if ck == 0xFFFFFFFF:
                    continue
                r = (ck >> 16) & 0xFF
                g = (ck >> 8) & 0xFF
                b = ck & 0xFF
                # Пропускаем фон, если так настроено
                if ignore_background and bg_rgb is not None and (r, g, b) == tuple(map(int, bg_rgb)):
                    continue
                x = int(xs)
                wseg = int(xe - xs)
                gid = 0
                if labels is not None:
                    gid = int(labels[y, x])
                key = (r, g, b, x, wseg, gid)
                present_keys.add(key)
                if key in open_rects:
                    rect = open_rects[key]
                    rect[3] += 1  # h += 1
                else:
                    # Новый прямоугольник: [x, y0, w, h]
                    open_rects[key] = [x, y, wseg, 1]

            # Закрываем прямоугольники, которые не продлились на текущей строке
            to_close = [k for k in open_rects.keys() if k not in present_keys]
            for k in to_close:
                r, g, b, x, wseg, gid = k
                x0, y0, width_r, height_r = open_rects[k]
                rect_elem = dwg.rect(insert=(x0, y0), size=(width_r, height_r), fill=rgb_to_hex((r, g, b)), stroke='none')
                if labels is not None and gid > 0:
                    gkey = f"obj-{gid}"
                    if gkey not in groups:
                        groups[gkey] = dwg.g(id=gkey)
                        dwg.add(groups[gkey])
                    if bring_black_to_front and is_black((r, g, b), black_threshold):
                        black_pending.append(rect_elem)
                    else:
                        groups[gkey].add(rect_elem)
                else:
                    if bring_black_to_front and is_black((r, g, b), black_threshold):
                        black_pending.append(rect_elem)
                    else:
                        dwg.add(rect_elem)
                del open_rects[k]

        # Закрываем все оставшиеся прямоугольники
        for k, (x0, y0, width_r, height_r) in open_rects.items():
            r, g, b, _, _, gid = k
            rect_elem = dwg.rect(insert=(x0, y0), size=(width_r, height_r), fill=rgb_to_hex((r, g, b)), stroke='none')
            if labels is not None and gid > 0:
                gkey = f"obj-{gid}"
                if gkey not in groups:
                    groups[gkey] = dwg.g(id=gkey)
                    dwg.add(groups[gkey])
                if bring_black_to_front and is_black((r, g, b), black_threshold):
                    black_pending.append(rect_elem)
                else:
                    groups[gkey].add(rect_elem)
            else:
                if bring_black_to_front and is_black((r, g, b), black_threshold):
                    black_pending.append(rect_elem)
                else:
                    dwg.add(rect_elem)

    if method == "contours":
        render_by_contours()
    else:
        if connect_same_color and float(connect_radius_px) > 0.0:
            # Режим «соединения объектов одного цвета в радиусе N пикселей»
            r = float(connect_radius_px)
            if r < 1.0:
                kernel_size = 1  # «радиус 0.5» эквивалентен отсутствию мостов через 1px
            else:
                kernel_size = int(r) * 2 + 1  # 1->3, 2->5, консервативно (floor)
            if bool(connect_4neighbors):
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
            else:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            processed = 0
            for (r, g, b, a), area in zip(unique_rgba, counts):
                if a == 0:
                    continue
                rgb = (int(r), int(g), int(b))
                if ignore_background and bg_rgb is not None and tuple(rgb) == tuple(bg_rgb):
                    continue
                if area < max(1, int(min_area)):
                    continue
                # Маска нужного цвета
                sel = (
                    (qrgb[:, :, 0] == r)
                    & (qrgb[:, :, 1] == g)
                    & (qrgb[:, :, 2] == b)
                    & (img[:, :, 3] > 0)
                )
                if not np.any(sel):
                    continue
                m = sel.astype(np.uint8) * 255
                # Выбор операции соединения:
                # - "close": морфологическое закрытие — аккуратно мостит зазоры <= радиуса (рекомендуется для 1px)
                # - "dilate": простое расширение — может сцепить объекты на расстоянии до 2*радиуса
                # - "none": без изменений — не соединять вовсе
                cop = str(connect_op).lower().strip()
                if cop == "dilate":
                    m = cv2.dilate(m, kernel, iterations=1)
                elif cop == "close":
                    # По умолчанию — close
                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
                else:
                    # none / неизвестное — оставляем маску как есть
                    pass
                paths_d = contours_to_evenodd_path(
                    m,
                    simplify_epsilon=0.0,
                    enable_smoothing=enable_smoothing,
                    snap_90deg_corners=snap_90deg_corners,
                    snap_90_one_pixel_only=snap_90_one_pixel_only,
                    stair2_remove_interior=stair2_remove_interior,
                    stair2_min_len=stair2_min_len,
                    stair2_remove_exterior=stair2_remove_exterior,
                    stair2_exterior_min_len=stair2_exterior_min_len,
                    stair2_exterior_only_outer=stair2_exterior_only_outer,
                    stair3_delete_by_first=stair3_delete_by_first,
                    stair3_min_edges=stair3_min_edges,
                    stair3_max_step_px=stair3_max_step_px,
                    stair3_policy=stair3_policy,
                    simplify_collinear=simplify_collinear,
                    collinear_tol=collinear_tol,
                    stair4_enable=stair4_enable,
                    stair4_target_angle_deg=stair4_target_angle_deg,
                    stair4_angle_tol_deg=stair4_angle_tol_deg,
                    stair4_only_internal=stair4_only_internal,
                    stair4_min_polygon_vertices=stair4_min_polygon_vertices,
                    ra90_enable=ra90_enable,
                    ra90_max_edge_len_px=ra90_max_edge_len_px,
                    ra90_min_edges=ra90_min_edges,
                    avoid_overlapping_vertices=avoid_overlapping_vertices,
                    simplify_stair_triplets=simplify_stair_triplets,
                    stair_max_distance_px=stair_max_distance_px,
                    stair_action=stair_action,
                    stair_collapse_runs=stair_collapse_runs,
                    stair_run_min_len=stair_run_min_len,
                    stair_run_keep_lowest_only=stair_run_keep_lowest_only,
                )
                if not paths_d:
                    continue
                fill = rgb_to_hex(rgb)
                for (d, centroid) in paths_d:
                    path = dwg.path(d=d, fill=fill, stroke='none')
                    path.attribs['fill-rule'] = 'evenodd'
                    path.attribs['shape-rendering'] = 'crispEdges'
                    if is_black(rgb, black_threshold):
                        if not place_path_in_black_region(path, centroid, rgb):
                            if bring_black_to_front:
                                black_paths_pending.append(path)
                            else:
                                dwg.add(path)
                    else:
                        if not place_path_in_black_region(path, centroid, rgb):
                            dwg.add(path)
                processed += 1
                if processed % 20 == 0:
                    print(f"  Обработано цветов (connect): {processed}/{len(unique_rgba)}")
        else:
            render_by_rle()

    # В самом конце формируем единый верхний слой с чёрными элементами,
    # чтобы ничто их не перекрывало: регионы, отдельные пути и (из RLE) прямоугольники
    # Собираем итоговые группы регионов: единая группа на регион,
    # внутри — сначала содержимое, затем чёрные элементы (чтобы чёрные были поверх)
    overlay_elems = []
    if black_regions_data:
        for reg in black_regions_data:
            has_any = reg.get('has_content') or reg.get('has_black')
            if not has_any:
                continue
            region_group = dwg.g()
            # порядок важен: сперва цветной контент, затем чёрные элементы
            region_group.add(reg['content_group'])
            region_group.add(reg['black_group'])
            overlay_elems.append(region_group)

    if bring_black_to_front and (overlay_elems or black_paths_pending or black_pending):
        overlay = dwg.g(id="overlay-black-top")
        for g in overlay_elems:
            overlay.add(g)
        for p in black_paths_pending:
            overlay.add(p)
        for el in black_pending:
            overlay.add(el)
        dwg.add(overlay)
    else:
        # Если не требуется переносить наверх, просто добавим непустые регионы в обычный поток
        for g in overlay_elems:
            dwg.add(g)

    # 6) Опциональный OCR (не влияет на геометрию)
    if ocr and _TESS and pytesseract is not None:
        if tesseract_cmd:
            try:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            except Exception:
                pass
        try:
            pil_rgb = image.convert('RGB')
            np_gray = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2GRAY)
            # Небольшое повышение контраста для лучшего OCR
            np_gray = cv2.normalize(np_gray, None, 0, 255, cv2.NORM_MINMAX)
            cfg = r"--oem 3 --psm 6"
            data = pytesseract.image_to_data(np_gray, config=cfg, output_type=pytesseract.Output.DICT)
            n = len(data['level'])
            for i in range(n):
                txt = str(data['text'][i]).strip()
                if not txt:
                    continue
                x, y, w_box, h_box = (
                    int(data['left'][i]),
                    int(data['top'][i]),
                    int(data['width'][i]),
                    int(data['height'][i]),
                )
                # Размещаем текст по левому нижнему углу контейнера
                dwg.add(
                    dwg.text(
                        txt,
                        insert=(x, y + h_box - 2),
                        font_size=h_box,
                        fill="#000000",
                    )
                )
        except Exception as e:
            print(f"OCR пропущен: {e}")
    elif ocr and not _TESS:
        print("Внимание: pytesseract недоступен. Установите Tesseract OCR и пакет pytesseract.")

    # 7) Сохраняем SVG
    try:
        dwg.save()
        print(f"SVG сохранён: {output_svg}")
    except Exception as e:
        print(f"Ошибка сохранения SVG: {e}")


# -----------------------------
# GUI ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------

def select_file() -> str:
    Tk().withdraw()
    return askopenfilename(filetypes=(("BMP files", "*.bmp"), ("All files", "*.*")))


def select_save_path() -> str:
    Tk().withdraw()
    return asksaveasfilename(defaultextension=".svg", filetypes=(("SVG files", "*.svg"),))


# -----------------------------
# ЕДИНЫЙ БЛОК НАСТРОЕК (редактируйте параметры здесь)
# -----------------------------
# Все параметры, кроме путей к файлам, собраны в одном словаре CFG.
# Меняйте значения ниже — вызов в main использует **CFG.
CFG = {
    # Геометрия контуров: 0 — без упрощения для максимальной точности.
    "simplify_epsilon": 0.0,

    # Минимальная площадь компоненты (в пикселях), меньше — игнорируется.
    "min_area": 1,

    # Игнорировать фон (не дублировать фоновые пиксели как фигуры).
    "ignore_background": True,

    # Рисовать фон в SVG как сплошной прямоугольник первым слоем.
    "draw_background": True,

    # Слияние близких цветов (шаг квантования 1 = без изменений; 2–6 сгладит ореолы антиалиасинга).
    "color_merge_step": 1,

    # Метод рендеринга: "rle" — пиксельно-точно (больше элементов), "contours" — по контурам (меньше узлов).
    "method": "rle",

    # Масштаб входного изображения (NEAREST). 2.0 означает: каждый исходный пиксель станет блоком 2x2 пикселя.
    "pre_scale_factor": 3.0,

    # Группировать объекты по 8-связности в <g id="obj-#">.
    "group_objects": True,

    # Минимальная площадь группы (в пикселях), меньше — отбрасывается при группировке.
    "min_group_area": 100,

    # Исключать «трубы/линии» по HSV и геометрии (тонкие и вытянутые). По умолчанию выключено.
    "exclude_pipes_by_hsv": False,
    # Нижняя/верхняя границы HSV-цвета труб (если exclude_pipes_by_hsv=True).
    "pipe_hsv_low": (80, 80, 30),
    "pipe_hsv_high": (140, 255, 255),
    # Порог «тонкости» и минимального соотношения сторон для исключения труб.
    "thin_max_width": 3,
    "thin_min_aspect": 4.0,

    # Шаг 1: «убрать лесенку» — замкнуть зазоры до 1px морфологическим закрытием (3x3), 8-связность.
    "bridge_gaps": True,

    # Соединять объекты одного цвета в радиусе N пикселей; при N<1 (например, 0.5) фактически ядро 1x1 — мостов нет.
    "connect_same_color": True,
    "connect_radius_px": 1.0,
    # Операция соединения фрагментов одного цвета:
    #   - "close" (по умолчанию): морфологическое закрытие — мостит ровно узкие зазоры (1px при радиусе=1)
    #   - "dilate": простая дилатация — может склеить объекты на расстоянии до 2*радиуса
    "connect_op": "dilate",
    # Тип соседства для ядра: False — 8-связность (квадрат 3x3), True — 4-соседство (крест 3x3)
    "connect_4neighbors": True,

    # Пост-упрощение: удалять узлы, лежащие на одной прямой (коллинеарные)
    # collinear_tol=0.0 — строго; >0 допускает небольшие отклонения (единицы пикселя)
    # Глобальный переключатель сглаживания лесенок (не влияет на коллинеарные)
    # Включаем сглаживание лесенок: триплеты и доп. проходы
    "enable_smoothing": False,
    # Снап 90°: заменять пару 45°-поворотов в одну сторону одним узлом на внешнем угле пикселя
    "snap_90deg_corners": True,
    # Ограничивать снап только на «однопиксельных» диагоналях (|dx|=|dy|=1). Выключите, если хотите снапить и более длинные диагонали.
    # Разрешаем снап 90° и для диагоналей > 1 пикселя, чтобы схлопывать больше случаев
    "snap_90_one_pixel_only": False,
    # Новый алгоритм: в чередующихся сериях H,V,H,V удалять все внутренние повороты
    "stair2_remove_interior": False,
    # Минимальная длина серии (в рёбрах). 3 = HVH (две смены направления)
    "stair2_min_len": 3,
    # stair4: удалить внутренние вершины с углом ~135° (после локальной чистки коллинеарных)
    "stair4_enable": True,
    "stair4_target_angle_deg": 135.0,
    "stair4_angle_tol_deg": 15.0,
    "stair4_only_internal": True,
    "stair4_min_polygon_vertices": 5,
    # 8) ra90: последовательности прямых углов с короткими рёбрами
    "ra90_enable": True,
    "ra90_max_edge_len_px": 10.0,
    "ra90_min_edges": 5,
    # Параметры stair3 (алгоритм отключён; параметры игнорируются)
    "stair3_delete_by_first": False,
    # Минимум рёбер в серии для активации этого правила (HVHV = 4)
    "stair3_min_edges": 4,
    # Лимит длины шага для stair3 (не используется, т.к. stair3 отключён)
    "stair3_max_step_px": 20.0,
    "simplify_collinear": True,
    "collinear_tol": 0.0,
    # Убирать тройки "лесенки": (вправо-вниз, вправо, вправо-вниз) — DR,R,DR
    # Включаем обработку одиночных/редких лесенок (триплеты)
    "simplify_stair_triplets": False,
    # Доп. ограничение: применять удаление лесенки только если расстояние
    # между крайними точками шаблона не превышает порог (в пикселях).
    # None — без ограничения. По умолчанию 6.0 px.
    # Порог дистанции между крайними точками триплета — ослабим (по умолчанию 6px)
    "stair_max_distance_px": 7.0,
    # Действие для лесенки: "average" — усреднить p1..p3; "remove13" — удалить p1 и p3 (p2 оставить)
    "stair_action": "average",
    # Коллапс серий лесенок, идущих в одном направлении (без закруглений):
    # при включении — удаляем все тройки в серии, кроме самой нижней, где сохраняем самую нижнюю точку
    # Коллапс серий лесенок в одном направлении — включим для более агрессивного сглаживания
    "stair_collapse_runs": False,
    # Минимальная длина серии (в количестве подряд идущих троек), чтобы применять коллапс
    "stair_run_min_len": 2,
    # В нижней тройке сохранять только самую нижнюю точку (остальные удалить)
    "stair_run_keep_lowest_only": False,
    # Удалять подряд идущие одинаковые вершины (защита от наложения узлов)
    "avoid_overlapping_vertices": False,

    # В конце выводить все чёрные фигуры поверх остальных (оверлей).
    "bring_black_to_front": True,
    # Порог близости к чёрному: каналы R,G,B <= threshold считаются чёрными.
    "black_threshold": 8,

    # OCR: распознавать текст (не влияет на геометрию), по умолчанию выключен.
    "ocr": False,
    # Путь к tesseract.exe (Windows). Оставьте None, если в PATH или OCR выключен.
    "tesseract_cmd": None,
}


if __name__ == "__main__":
    print("bmp2svg (точная векторизация) — запуск")
    src = select_file()
    if not src:
        print("Файл не выбран — выход")
        sys.exit(0)
    dst = select_save_path()
    if not dst:
        print("Путь сохранения не выбран — выход")
        sys.exit(0)

    # Запуск с централизованными настройками — редактируйте словарь CFG выше
    bmp_to_svg_precise(src, dst, **CFG)
