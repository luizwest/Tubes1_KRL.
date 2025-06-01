from typing import Optional
from game.logic.base import BaseLogic
from game.models import GameObject, Board, Position
from ..util import get_direction

# Kelas utama logika bot yang mengimplementasikan strategi greedy dengan bobot dinamis
class KRL(BaseLogic):
    def __init__(self):
        # Arah gerak: kanan, bawah, kiri, atas
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.goal_position: Optional[Position] = None
        self.current_direction = 0
        self.board_width = 15
        self.board_height = 15

    # Menghitung jarak Manhattan antara dua titik
    def manhattan_distance(self, pos1: Position, pos2: tuple) -> int:
        return abs(pos1.x - pos2[0]) + abs(pos1.y - pos2[1])

    # Mengecek apakah posisi (x, y) masih berada dalam papan permainan
    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.board_width and 0 <= y < self.board_height

    # Menghitung bobot prioritas untuk setiap aksi berdasarkan kondisi permainan
    def compute_priority_weights(self, board_bot: GameObject, board: Board):
        props = board_bot.properties
        pos = board_bot.position
        base = props.base

        weight_return_base = 0
        weight_attack = 0
        weight_collect = 0
        weight_button = 0

        # Bobot untuk kembali ke base berdasarkan jumlah diamond
        if props.diamonds >= 4:
            weight_return_base += 10
        elif props.diamonds == 3:
            weight_return_base += 6
        elif props.diamonds == 2:
            weight_return_base += 3

        # Bobot untuk menyerang musuh di sekitar
        for p in board.bots:
            if p != board_bot and p.properties.diamonds > 0:
                dist = self.manhattan_distance(pos, (p.position.x, p.position.y))
                if dist <= 2:
                    weight_attack += 5 - dist

        # Bobot untuk mengumpulkan diamond
        diamond_count = len([obj for obj in board.game_objects if obj.type == "DiamondGameObject"])
        if diamond_count >= 6:
            weight_collect += 8
        elif diamond_count >= 3:
            weight_collect += 5
        else:
            weight_collect += 2

        # Bobot untuk menekan tombol merah jika diamond di papan sedikit
        red_button = self.get_red_button(board)
        if red_button and diamond_count < 4:
            nearest_diamond = self.get_closest_diamond(board_bot, board)
            dist_button = self.manhattan_distance(pos, (red_button.position.x, red_button.position.y))
            dist_diamond = self.manhattan_distance(pos, (nearest_diamond.position.x, nearest_diamond.position.y)) if nearest_diamond else 99
            if dist_button < dist_diamond:
                weight_button += 7

        return {
            'return': weight_return_base,
            'attack': weight_attack,
            'collect': weight_collect,
            'button': weight_button
        }

    # Mendapatkan diamond terdekat dari posisi bot
    def get_closest_diamond(self, bot: GameObject, board: Board):
        diamonds = [d for d in board.diamonds]
        if not diamonds:
            return None
        return min(diamonds, key=lambda d: self.manhattan_distance(bot.position, (d.position.x, d.position.y)))

    # Mendapatkan tombol merah (jika ada)
    def get_red_button(self, board: Board):
        for obj in board.game_objects:
            if obj.type == "DiamondButtonGameObject":
                return obj
        return None

    # Menentukan target terbaik berdasarkan prioritas aksi
    def find_best_target(self, board_bot: GameObject, board: Board) -> Optional[Position]:
        props = board_bot.properties
        pos = board_bot.position
        base = props.base

        if props.diamonds > 0 and self.manhattan_distance(pos, (base.x, base.y)) == 1:
            return base

        # Hitung bobot prioritas
        weights = self.compute_priority_weights(board_bot, board)
        priorities = sorted(weights.items(), key=lambda x: -x[1])

        # Pilih aksi dengan prioritas tertinggi
        for action, _ in priorities:
            if action == 'return' and props.diamonds >= 2:
                return base

            elif action == 'attack' and props.diamonds < 5:
                opponents = [p for p in board.bots if p != board_bot and p.properties.diamonds > 0]
                if opponents:
                    nearest = min(opponents, key=lambda p: self.manhattan_distance(pos, (p.position.x, p.position.y)))
                    if self.manhattan_distance(pos, (nearest.position.x, nearest.position.y)) <= 2:
                        return nearest.position

            elif action == 'collect':
                diamond = self.get_closest_diamond(board_bot, board)
                if diamond:
                    return diamond.position

            elif action == 'button':
                button = self.get_red_button(board)
                if button:
                    return button.position

        # Jika semua gagal, gunakan teleporter jika menguntungkan
        teleporters = [obj for obj in board.game_objects if obj.type == "TeleporterGameObject"]
        diamonds = [obj for obj in board.game_objects if obj.type == "DiamondGameObject"]
        if teleporters and diamonds:
            return self.evaluate_teleporters(board_bot, teleporters, diamonds, base)

        return base

    # Evaluasi apakah menggunakan teleporter lebih efektif
    def evaluate_teleporters(self, board_bot: GameObject, teleporters, diamonds, base):
        if len(teleporters) < 2:
            return None

        t1, t2 = teleporters
        t1_pos = (t1.position.x, t1.position.y)
        t2_pos = (t2.position.x, t2.position.y)

        nearest_diamond = min(diamonds, key=lambda d: self.manhattan_distance(board_bot.position, (d.position.x, d.position.y)))
        diamond_pos = (nearest_diamond.position.x, nearest_diamond.position.y)

        dist_to_diamond = self.manhattan_distance(board_bot.position, diamond_pos)
        dist_t1 = self.manhattan_distance(board_bot.position, t1_pos)
        dist_t2 = self.manhattan_distance(board_bot.position, t2_pos)
        dist_after_t1 = self.manhattan_distance(Position(t2_pos[0], t2_pos[1]), diamond_pos)
        dist_after_t2 = self.manhattan_distance(Position(t1_pos[0], t1_pos[1]), diamond_pos)

        if self.manhattan_distance(board_bot.position, (base.x, base.y)) <= 2:
            return None

        if dist_t1 + dist_after_t1 + 1 < dist_to_diamond:
            return t1.position
        elif dist_t2 + dist_after_t2 + 1 < dist_to_diamond:
            return t2.position

        return None

    # Mencari musuh di sekitar untuk ditackle
    def find_enemy_in_range_to_tackle(self, board_bot: GameObject, board: Board):
        if board_bot.properties.diamonds >= 5:
            return None

        pos = board_bot.position
        for dx, dy in self.directions:
            new_x, new_y = pos.x + dx, pos.y + dy
            for other_bot in board.bots:
                if other_bot != board_bot and other_bot.properties.diamonds > 0:
                    if other_bot.position.x == new_x and other_bot.position.y == new_y:
                        return dx, dy
        return None

    # Menghindari musuh yang terlalu dekat
    def avoid_adjacent_enemies(self, board_bot: GameObject, board: Board):
        pos = board_bot.position
        for dx, dy in self.directions:
            nx, ny = pos.x + dx, pos.y + dy
            for other_bot in board.bots:
                if other_bot != board_bot and other_bot.position.x == nx and other_bot.position.y == ny:
                    for adx, ady in self.directions:
                        alt_x, alt_y = pos.x + adx, pos.y + ady
                        if self.is_valid_position(alt_x, alt_y):
                            if all(ob.position.x != alt_x or ob.position.y != alt_y for ob in board.bots):
                                return adx, ady
        return None

    # Fungsi utama untuk menentukan gerakan bot pada setiap turn
    def next_move(self, board_bot: GameObject, board: Board):
        # Coba tackle musuh jika bisa
        tackle_dir = self.find_enemy_in_range_to_tackle(board_bot, board)
        if tackle_dir:
            return tackle_dir

        # Hindari musuh di sekitar
        avoid_dir = self.avoid_adjacent_enemies(board_bot, board)
        if avoid_dir:
            return avoid_dir

        # Menuju target terbaik
        goal_position = self.find_best_target(board_bot, board)
        if goal_position:
            delta_x, delta_y = get_direction(
                board_bot.position.x, board_bot.position.y,
                goal_position.x, goal_position.y
            )
            if not self.is_valid_position(board_bot.position.x + delta_x, board_bot.position.y + delta_y):
                delta_x, delta_y = 0, 0
        else:
            delta_x, delta_y = 0, 0

        return delta_x, delta_y
