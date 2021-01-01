import os
import tkinter as tk

import time
from PIL import ImageTk, Image

from ws.RLUtils.common.misc_functions import calc_pixels
from ..logic.SETUP_INFO import ACTION_MOVE_STATE_RULES
from ws.RLEnvironments.gridworld.grid_board.qwaste_mgt import qwaste_mgt

PhotoImage = ImageTk.PhotoImage

COORD_LEFT = (7, 42)  # left
COORD_RIGHT = (77, 42)  # right
COORD_UP = (42, 5)  # up
COORD_DOWN = (42, 77)  # down

class display_mgt:
    def __init__(self, app_info):
        self._tk = tk.Tk()
        self._cursor = None
        app_fn_display_info = app_info["display"]
        self._unit = app_fn_display_info["UNIT"]
        self._width = app_fn_display_info["WIDTH"]
        self._height = app_fn_display_info["HEIGHT"]
        self._board_blockers = app_fn_display_info["BOARD_BLOCKERS"]
        self._board_goal = app_fn_display_info["BOARD_GOAL"]
        # self._fnQWasteDestructiveGet, self._fnQWastePushIfEmpty = qwaste_mgt()
        self._right_margin = 5
        self._bottom_margin = 80
        self._tk.title(app_info["display"]["APP_NAME"])
        self._fnQWasteDestructiveGet, self._fnQWastePushIfEmpty = qwaste_mgt()

    def fnInit(self, acton_dictionary):
        self._tk.geometry('{0}x{1}'.format(self._width * self._unit + self._right_margin,
                                           self._height * self._unit + self._bottom_margin))
        self._tk.texts = []
        self._tk.arrows = []

        (self._tk.up, self._tk.down, self._tk.left, self._tk.right), self._tk.shapes = self._fn_load_images()
        self._tk.canvas = self._fn_build_canvas(acton_dictionary)

        self._fn_append_rewards_to_canvas()
        self._fn_render_on_canvas()

        self._tk.mainloop()

    def fn_move_cursor(self, stateStart, stateEnd=(0, 0)):
        step = self._fn_calculate_step(stateStart, stateEnd)

        self._tk.canvas.move(self._cursor, step[0] * self._unit, step[1] * self._unit)
        self._fn_render_on_canvas()

    def fn_show_policy_arrows(self, policy_table):
        for i in self._tk.arrows:
            self._tk.canvas.delete(i)

        for i in range(self._height):
            for j in range(self._width):
                self._fn_draw_arrow(i, j, policy_table[i][j])

    def fn_show_state_values(self, value_table):
        for i in self._tk.texts:
            self._tk.canvas.delete(i)

        for i in range(self._height):
            for j in range(self._width):
                val = round(value_table[i][j], 8)
                self._fn_append_text_canvas(i, j, val)
        self._fn_append_rewards_to_canvas()
        self._fn_render_on_canvas()

    def fn_show_qvalue(self, state, q_actions):
        stateStr = str(state)
        wasteVal = self._fnQWasteDestructiveGet(stateStr)
        if wasteVal is not None:
            for i in wasteVal:
                self._tk.canvas.delete(i)

        q_action_list = list(q_actions)
        refs = [self._fn_show_qvalue_directions(state, q_action_list[0], COORD_UP),
                self._fn_show_qvalue_directions(state, q_action_list[1], COORD_DOWN),
                self._fn_show_qvalue_directions(state, q_action_list[2], COORD_LEFT),
                self._fn_show_qvalue_directions(state, q_action_list[3], COORD_RIGHT)]

        success = self._fnQWastePushIfEmpty(stateStr, refs)
        if not success:
            for i in refs:
                self._tk.canvas.delete(i)
        self._fn_append_rewards_to_canvas()
        self._fn_render_on_canvas()

    # PRIVATE METHODS
    @staticmethod
    def _fn_calculate_step(state, newState):
        stepX, stepY = newState[0] - state[0], newState[1] - state[1]
        return stepX, stepY

    def _fn_show_qvalue_directions(self, state, stateAction, coord):
        x = coord[0] + self._unit * state[0]
        y = coord[1] + self._unit * state[1]
        val = round(stateAction, 2)
        font = ('Helvetica', str(10), 'normal')
        text_ref = self._tk.canvas.create_text(x, y, fill="black", text=str(val),
                                               font=font, anchor="nw")
        return text_ref

    def _fn_append_rewards_to_canvas(self):
        self._fn_append_reward_to_canvas(self._board_goal['x'], self._board_goal['y'], str(self._board_goal['reward']))
        for blocker in self._board_blockers:
            self._fn_append_reward_to_canvas(blocker['x'], blocker['y'], str(blocker['reward']))

    def _fn_append_reward_to_canvas(self, row, col, contents, font='Helvetica', size=10,
                             style='bold', anchor="nw"):
        origin_x, origin_y = 45, 50
        x, y = origin_x + (self._unit * row), origin_y + (self._unit * col)
        font = (font, str(size), style)
        text = self._tk.canvas.create_text(x, y, fill="yellow", text=contents,
                                           font=font, anchor=anchor)
        self._tk.texts.append(text)

    def _fn_render_on_canvas(self):
        time.sleep(0.1)
        self._tk.canvas.tag_raise(self._cursor)
        self._tk.update()

    def _fn_create_button(self, canvas, button_x_offset, button_name, button_action):

        bound_button = tk.Button(bg="gray",
                                 text=button_name,
                                 command=button_action)
        bound_button.configure(width=10, height=2)
        canvas.create_window(self._width * self._unit * button_x_offset, self._height * self._unit + 23,
                             window=bound_button)

    def _fn_build_canvas(self, acton_dictionary):
        canvas = tk.Canvas(
            height=self._height * self._unit,
            width=self._width * self._unit)
        button_x_offset = .10
        for label, fn in acton_dictionary.items():
            self._fn_create_button(canvas, button_x_offset, label, fn)
            button_x_offset += .20

        # create lines
        for col in range(0, (self._width + 1) * self._unit, self._unit):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, self._height * self._unit
            canvas.create_line(x0, y0, x1, y1)

        for row in range(0, (self._height + 1) * self._unit, self._unit):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, self._width * self._unit, row
            canvas.create_line(x0, y0, x1, y1)

        self._cursor = canvas.create_image(self._unit / 2, self._unit / 2, image=self._tk.shapes[2])
        for blocker in self._board_blockers:
            pix_x, pix_y = calc_pixels(self._unit, blocker['x'], blocker['y'])
            canvas.create_image(pix_x, pix_y, image=self._tk.shapes[0])

        pix_x, pix_y = calc_pixels(self._unit, self._board_goal['x'], self._board_goal['y'])
        canvas.create_image(pix_x, pix_y, image=self._tk.shapes[1])
        canvas.pack()
        return canvas

    @staticmethod
    def _fn_load_images():
        rwd = os.path.dirname(__file__)
        image_dir = os.path.join(rwd, 'img')

        up = PhotoImage(Image.open(image_dir + "/up.png").resize((13, 13)))
        right = PhotoImage(Image.open(image_dir + "/right.png").resize((13, 13)))
        left = PhotoImage(Image.open(image_dir + "/left.png").resize((13, 13)))
        down = PhotoImage(Image.open(image_dir + "/down.png").resize((13, 13)))
        rectangle = PhotoImage(Image.open(image_dir + "/penalty_box.png").resize((65, 65)))
        triangle = PhotoImage(Image.open(image_dir + "/reward_box.png").resize((65, 65)))
        circle = PhotoImage(Image.open(image_dir + "/cursor.png").resize((32, 32)))
        return (up, down, left, right), (rectangle, triangle, circle)

    def _fn_append_text_canvas(self, col, row, contents, font='Helvetica', size=10,
                           style='normal', anchor="nw"):
        origin_x, origin_y = 10, 85
        x, y = origin_x + (self._unit * row), origin_y + (self._unit * col)
        font = (font, str(size), style)
        text = self._tk.canvas.create_text(x, y, fill="black", text=contents,
                                           font=font, anchor=anchor)
        self._tk.texts.append(text)

    def _fn_draw_arrow(self, col, row, policy):

        if col == self._board_goal['y'] and row == self._board_goal['x']:
            return

        if policy[0] > 0:  # up
            origin_x, origin_y = 50 + (self._unit * row), 10 + (self._unit * col)
            self._tk.arrows.append(self._tk.canvas.create_image(origin_x, origin_y,
                                                                image=self._tk.up))
        if policy[1] > 0:  # down
            origin_x, origin_y = 50 + (self._unit * row), 90 + (self._unit * col)
            self._tk.arrows.append(self._tk.canvas.create_image(origin_x, origin_y,
                                                                image=self._tk.down))
        if policy[2] > 0:  # left
            origin_x, origin_y = 10 + (self._unit * row), 50 + (self._unit * col)
            self._tk.arrows.append(self._tk.canvas.create_image(origin_x, origin_y,
                                                                image=self._tk.left))
        if policy[3] > 0:  # right
            origin_x, origin_y = 90 + (self._unit * row), 50 + (self._unit * col)
            self._tk.arrows.append(self._tk.canvas.create_image(origin_x, origin_y,
                                                                image=self._tk.right))

    def fn_is_target_state_reached(self, state):
        if state == (self._board_goal['x'], self._board_goal['y']):
            return True
        return False

    @staticmethod
    def fn_get_start_state():
        return [0, 0]

    @staticmethod
    def fn_run_next_move(state, fnNextGetAction):
        action = fnNextGetAction(state)
        if action < 0:
            return None

        next_state = ACTION_MOVE_STATE_RULES[action]
        new_x = state[0] + next_state[0]
        new_y = state[1] + next_state[1]
        return new_x, new_y
